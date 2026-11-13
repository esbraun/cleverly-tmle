r"""The property rows of ``asymmetric-wald``, their summary, verdicts, witnesses and falsifiers.

Each group of :data:`tests.studies.asymmetric_wald.GROUPS` is one draw per replicate, keyed
``(group, replicate)`` and drawn from ``stream_seed(STUDY, "property_sample", group,
replicate)``.  Every cell of the draw's fit writes one row
(:func:`~tests.studies.asymmetric_wald.rule_columns`): the shared columns with the plug-in
standard error, PR 3's interval, the small-sample columns, the acceleration and a difference's
arm estimates with their implied correlation, so every candidate interval is recomputed from
the rows (:func:`candidate_interval`).  ``covered`` is 1 when the rule's interval is shown and
covers the truth: a withheld interval is a miss.  ``rejected`` is ``1 - covered``.

Each gated cell also writes one ``control`` row per control of
:func:`~tests.studies.asymmetric_wald.controls_of`, the same fit read under the control's
interval with the same withheld trigger.  Controls are reported (:data:`DIAGNOSTIC`).

**The gated reading** (plan rev 6, 9.5; :func:`summarize_properties`):

* *coverage legs on the first 2,000 replicates* by index: the lower leg's 99% Clopper-Pearson
  lower end, withheld and lost fits as misses over the 2,000, at least 0.92; the shown-only
  upper end at most 0.98;
* *tail clause on all 4,000*: for each side, the 99% CP lower end of the count of shown fits
  whose interval lies entirely below (above) the truth, over the shown fits, at most 0.040;
* on the owner cells, the *paired gain* over all 4,000: the exact conditional 99.5% one-sided
  lower bound of ``b / (b + c)`` above 1/2, against identity normal (MOVER) or the capped t
  (skew-widened).

A gated row's ``passed`` holds the complete case of the legs (the failure accounting turns it
into the worst case, study 1's D13) and of the tail clause and gain.  A reported row's
``passed`` holds the same reading as its own, with the declared legs.  Each witness is a
summary row: the source cell's fits under a mutated interval, ``passed`` when the mutation fails
its heavy-side tail clause.  ``falsifiers.csv`` (:func:`falsifiers`) evaluates the five declared
falsifiers of plan 9.6 on the held-out cells, including the C3 width clause
(:func:`width_ratio_bound`, R6-1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from tests.parallel import STUDY_JOBS
from tests.studies import asymmetric_wald as study
from tests.studies import asymmetric_wald_laws as heldout
from tests.studies import small_sample_laws as laws
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.properties import REPLICATE_COLUMNS, ratio_intervals
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS
DIAGNOSTIC = study.DIAGNOSTIC

#: The share of a cell's requested replicates the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The ungated falsifier table's file name.
FALSIFIERS_FILE = "falsifiers.csv"

#: Every column a property row carries.
ROW_COLUMNS = (*REPLICATE_COLUMNS, *study.EXTRA_PROPERTY_COLUMNS)

#: The candidate intervals every summary row reports, each read off the rows.
CANDIDATES = (
    "identity_normal",
    "capped_t",
    "range_logit_normal",
    "logit_t",
    "mover",
    "mover_rho0",
    "skew_t",
    "cf_shift",
)
#: The mutated candidates the witnesses read.
MUTATIONS = ("sign_flip", "mover_swap")

#: The candidate each control reads.
CONTROL_CANDIDATE = {
    "normal_reference_control": "identity_normal",
    "capped_t_control": "capped_t",
    "range_logit_control": "range_logit_normal",
    "logit_t_control": "logit_t",
    "cf_shift_control": "cf_shift",
}

#: The columns the summary publishes beside the shared ones.
SUMMARY_COLUMNS = (
    "shown_replicates",
    "shown_coverage",
    "shown_ci_lower",
    "shown_ci_upper",
    "leg_replicates",
    "leg_survivors",
    "leg_coverage",
    "leg_complete_ci_lower",
    "lower_leg_ci_lower",
    "lower_leg_ci_upper",
    "leg_shown",
    "leg_shown_coverage",
    "leg_shown_ci_upper",
    "below_count",
    "above_count",
    "below_ci_lower",
    "above_ci_lower",
    "tails_passed",
    "constant_cell_rate",
    "out_of_range_rate",
    "expected_withheld",
    "expected_failed",
    "lower_miss_rate",
    "upper_miss_rate",
    *(f"coverage_{round(level * 100):02d}" for level in study.REPORTED_LEVELS),
    "nu_median",
    "acceleration_median",
    "se_error_correlation",
    "oracle_sd_coverage",
    *(f"candidate_{name}" for name in CANDIDATES),
    "comparator",
    "gain_b",
    "gain_c",
    "gain_lower",
    "width_ratio",
    "width_ratio_upper",
)


# ------------------------------------------------------------------------------ intervals


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return frame[name].to_numpy(dtype=float)


def _quantile(df: np.ndarray, alpha: float) -> np.ndarray:
    level = 1.0 - alpha / 2.0
    with np.errstate(invalid="ignore"):
        return np.where(
            np.isinf(df), stats.norm.ppf(level), stats.t.ppf(level, np.where(np.isinf(df), 1.0, df))
        )


def _logit_interval(
    psi: np.ndarray, se: np.ndarray, low: np.ndarray, high: np.ndarray, quantile: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The Wald interval on the range-logit scale of ``(low, high)``; ``nan`` outside it."""
    with np.errstate(invalid="ignore", divide="ignore"):
        inside = np.isfinite(low) & np.isfinite(high) & (low < psi) & (psi < high)
        centre = np.log((psi - low) / (high - psi))
        spread = se * (high - low) / ((psi - low) * (high - psi))
        lower = low + (high - low) * expit(centre - quantile * spread)
        upper = low + (high - low) * expit(centre + quantile * spread)
    return np.where(inside, lower, np.nan), np.where(inside, upper, np.nan)


def _mover(
    frame: pd.DataFrame, quantile: np.ndarray, *, swap: bool = False, rho: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated MOVER on the arms' range-logit normal intervals; ``nan`` without arms."""
    p1, s1 = _column(frame, "arm1_estimate"), _column(frame, "arm1_std_error")
    p0, s0 = _column(frame, "arm0_estimate"), _column(frame, "arm0_std_error")
    zeros, ones = np.zeros(len(frame)), np.ones(len(frame))
    l1, u1 = _logit_interval(p1, s1, zeros, ones, quantile)
    l0, u0 = _logit_interval(p0, s0, zeros, ones, quantile)
    r = _column(frame, "arm_correlation") if rho else np.zeros(len(frame))
    down1, up1, down0, up0 = p1 - l1, u1 - p1, p0 - l0, u0 - p0
    if swap:
        down1, up1, down0, up0 = up1, down1, up0, down0
    d = p1 - p0
    with np.errstate(invalid="ignore"):
        lower = d - np.sqrt(np.maximum(down1**2 + up0**2 - 2.0 * r * down1 * up0, 0.0))
        upper = d + np.sqrt(np.maximum(up1**2 + down0**2 - 2.0 * r * up1 * down0, 0.0))
    return lower, upper


def candidate_interval(
    frame: pd.DataFrame, name: str, *, alpha: float = study.ALPHA
) -> tuple[np.ndarray, np.ndarray]:
    """One candidate interval of every row of ``frame``; ``nan`` where it does not apply.

    ``identity_normal``; ``capped_t`` at ``max(nu, 5)`` (a missing ``nu`` reads as the normal);
    ``range_logit_normal`` and ``logit_t`` on the row's bounds; ``mover`` (PR 3's difference
    rule), ``mover_rho0`` (without the correlation term) and ``mover_swap`` (each arm's sides
    swapped, a mutation) from the arm columns; ``skew_t`` (PR 3's continuous rule) and
    ``sign_flip`` (with ``-a``, a mutation) at the capped t; ``cf_shift``, the two-sided
    Cornish-Fisher shift at the capped t on a continuous row and ``z`` on a binary one; and
    ``rule``, each row's own construction.
    """
    psi, se = _column(frame, "estimate"), _column(frame, "std_error")
    nu = np.nan_to_num(_column(frame, "nu"), nan=math.inf)
    capped = np.maximum(nu, study.DF_CAP)
    normal = np.full(len(frame), math.inf)
    accel = _column(frame, "acceleration")
    low_b, high_b = _column(frame, "bound_low"), _column(frame, "bound_high")
    z = _quantile(normal, alpha)
    t = _quantile(capped, alpha)
    factor = 2.0 * z * z + 1.0
    if name == "rule":
        construction = frame["construction"].astype(str).to_numpy()
        out_low, out_high = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        for kind, source in (
            ("mover", "mover"),
            ("range_logit", "range_logit_normal"),
            ("skew_t", "skew_t"),
        ):
            mask = construction == kind
            if mask.any():
                low, high = candidate_interval(frame.loc[mask], source, alpha=alpha)
                out_low[mask], out_high[mask] = low, high
        return out_low, out_high
    if name == "identity_normal":
        return psi - z * se, psi + z * se
    if name == "capped_t":
        return psi - t * se, psi + t * se
    if name == "range_logit_normal":
        return _logit_interval(psi, se, low_b, high_b, z)
    if name == "logit_t":
        return _logit_interval(psi, se, low_b, high_b, t)
    if name in ("mover", "mover_rho0", "mover_swap"):
        return _mover(frame, z, swap=name == "mover_swap", rho=name != "mover_rho0")
    if name in ("skew_t", "sign_flip"):
        a = accel if name == "skew_t" else -accel
        shift = a * factor
        return psi - se * np.maximum(t, t - shift), psi + se * np.maximum(t, t + shift)
    if name == "cf_shift":
        gaussian = frame["outcome_family"].astype(str).to_numpy() == "gaussian"
        q = np.where(gaussian, t, z)
        shift = accel * factor
        return psi - se * (q - shift), psi + se * (q + shift)
    raise KeyError(name)


def withheld_triggers(frame: pd.DataFrame) -> np.ndarray:
    """Each row's withheld trigger, ``""`` for a shown interval (a null from Parquet too)."""
    return frame["withheld"].fillna("").astype(str).to_numpy()


def _covers(frame: pd.DataFrame, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Shown and covering; a withheld or undefined interval is a miss."""
    truth = _column(frame, "truth")
    shown = withheld_triggers(frame) == ""
    with np.errstate(invalid="ignore"):
        return shown & (low <= truth) & (truth <= high)


# ------------------------------------------------------------------------------ the rows


def _draw(label: str, replicate: int) -> pd.DataFrame:
    """The draw of replicate ``replicate`` of a group, from the stream keyed by its label."""
    group = study.GROUP_BY_LABEL[label]
    return study.sample(group.law, group.n, stream_seed(STUDY, "property_sample", label, replicate))


def _replicate(label: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every property row of one replicate of one group."""
    group = study.GROUP_BY_LABEL[label]
    truth = study.truths(group.law)
    result = study.fit_group(group, frame)
    rows: list[dict[str, Any]] = []
    for cell in group.cells:
        estimate = result[cell.estimand]
        reference = float(truth[cell.estimand])
        extra = study.rule_columns(result, cell.estimand, group.family)
        shown = extra["withheld"] == ""
        covered = int(shown and extra["rule_lower"] <= reference <= extra["rule_upper"])
        row = {
            "property": cell.family,
            "cell": cell.name,
            "role": cell.role,
            "replicate": replicate,
            "n": group.n,
            "requested_replicates": group.replicates,
            "failed_replicates": 0,
            "truth": reference,
            "estimate": float(estimate.psi),
            "std_error": float(estimate.plugin_std_error),
            "covered": covered,
            "rejected": 1 - covered,
            **extra,
        }
        rows.append(row)
        if cell.family not in study.COVERAGE_FAMILIES:
            continue
        one = pd.DataFrame([row])
        for control in study.controls_of(group, cell):
            low, high = candidate_interval(one, CONTROL_CANDIDATE[control])
            hit = int(_covers(one, low, high)[0])
            rows.append(
                {
                    **row,
                    "property": study.CONTROL,
                    "cell": f"{cell.name}__{control}",
                    "role": DIAGNOSTIC,
                    "covered": hit,
                    "rejected": 1 - hit,
                }
            )
    return rows


def _cells_of(key: tuple[str, int]) -> list[str]:
    group = study.GROUP_BY_LABEL[key[0]]
    out = []
    for cell in group.cells:
        out.append(f"{cell.family}/{cell.name}")
        if cell.family in study.COVERAGE_FAMILIES:
            out.extend(
                f"{study.CONTROL}/{cell.name}__{control}"
                for control in study.controls_of(group, cell)
            )
    return out


def _cell_of(key: tuple[str, int]) -> str:
    return f"group/{key[0]}"


def _replicate_of(key: tuple[str, int]) -> int:
    return key[1]


def payloads(
    budget: int | None = None, groups: Sequence[str] | None = None
) -> list[tuple[str, int]]:
    """Every ``(group, replicate)`` key in group order; ``budget`` caps, ``groups`` selects."""
    return [
        (group.label, index)
        for group in study.GROUPS
        if groups is None or group.label in groups
        for index in range(group.replicates if budget is None else min(budget, group.replicates))
    ]


def _fit_all(n_jobs: int, budget: int | None, groups: Sequence[str] | None) -> pd.DataFrame:
    outcomes = fit_properties(
        payloads(budget, groups),
        splat_drawn(_replicate),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        rows_at=None,
        seed_of=lambda key: stream_seed(STUDY, "property_sample", key[0], key[1]),
    )
    rows = pd.DataFrame([row for value in outcomes for row in value], columns=list(ROW_COLUMNS))
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    if groups is None:
        declared = {(family, cell) for family, cells in PROPERTY_CELLS.items() for cell in cells}
        declared -= {(study.WITNESS, cell) for cell in PROPERTY_CELLS[study.WITNESS]}
        found = set(zip(rows["property"], rows["cell"], strict=True))
        if found != declared:
            raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows


def generate_property_rows(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fit every property replicate; ``budget`` and ``groups`` restrict a pre-run check.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    budget : int or None
        A cap on each group's replicates; ``None`` is the declaration.
    groups : sequence of str or None
        The group labels to fit; ``None`` is every group.  A pre-run smoke passes
        :data:`~tests.studies.asymmetric_wald.SMOKE_GROUPS`, which hold no held-out law.

    Returns
    -------
    pandas.DataFrame
        The rows, :data:`ROW_COLUMNS`.
    """
    return _fit_all(n_jobs, budget, groups)


def generate_property_artifacts(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    groups: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every property replicate, and evaluate the falsifiers from the same rows.

    Parameters
    ----------
    n_jobs : int
        Worker processes.
    budget : int or None
        A cap on each group's replicates; ``None`` is the declaration.
    groups : sequence of str or None
        The group labels to fit; ``None`` is every group.

    Returns
    -------
    tuple
        The rows, and :data:`FALSIFIERS_FILE` by name.
    """
    rows = _fit_all(n_jobs, budget, groups)
    return rows, {FALSIFIERS_FILE: falsifier_table(rows)}


def stamped(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` with each cell's ``failed_replicates`` set to its missing replicates.

    The hook that writes :data:`FALSIFIERS_FILE` runs before the driver stamps the rows from
    the failure records.  A failed key costs every cell it writes one replicate, so a cell's
    gap is its failures, which is what the driver's stamp will record.
    """
    out = rows.copy()
    sizes = out.groupby(["property", "cell"])["replicate"].transform("size")
    out["failed_replicates"] = out["requested_replicates"] - sizes
    return out


def falsifier_table(rows: pd.DataFrame) -> pd.DataFrame:
    """:func:`falsifiers` of the summary of the rows as fitted, stamped by :func:`stamped`."""
    return falsifiers(summarize_properties(stamped(rows)), rows)


# ------------------------------------------------------------------------------ the summary


def _interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    iv = clopper_pearson(successes, trials, confidence_level=STUDY.margins.confidence_level)
    return iv.low, iv.high


def gain_lower(b: int, c: int, level: float = study.PAIRED_GAIN_LEVEL) -> float:
    """The exact conditional one-sided lower bound of ``b / (b + c)`` at ``level``."""
    return float(stats.beta.ppf(1.0 - level, b, c + 1)) if b > 0 else 0.0


def tail_lower(count: int, shown: int) -> float:
    """The 99% CP lower end of ``count`` one-sided misses among ``shown`` fits."""
    return _interval(count, shown)[0] if shown > 0 else math.nan


def tails_hold(below: int, above: int, shown: int) -> bool:
    """The tail clause: both one-sided rates' 99% CP lower ends at most the threshold."""
    return bool(
        shown > 0
        and tail_lower(below, shown) <= study.TAIL_THRESHOLD
        and tail_lower(above, shown) <= study.TAIL_THRESHOLD
    )


def width_ratio_bound(
    numerator: np.ndarray, denominator: np.ndarray, *, seed: int
) -> tuple[float, float]:
    """The mean width ratio and its 99% bootstrap upper bound (plan 9.6, nit R6-1).

    The ratio is ``sum(numerator) / sum(denominator)`` over the fits.  The bootstrap resamples
    fits with replacement, keeping each fit's two widths together,
    :attr:`~tests.studies.evidence.registry.Margins.bootstrap_replicates` times from ``seed``,
    and the bound is the percentile interval's upper end, the 0.995 quantile of the resampled
    ratios (the study's 99% two-sided convention).  Rows with a non-finite width are dropped.
    """
    keep = np.isfinite(numerator) & np.isfinite(denominator)
    top, bottom = numerator[keep], denominator[keep]
    if top.size < 2 or bottom.sum() <= 0:
        return math.nan, math.nan
    point = float(top.sum() / bottom.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(STUDY.margins.bootstrap_replicates)
    chunk = 500
    for start in range(0, draws.size, chunk):
        size = min(chunk, draws.size - start)
        index = rng.integers(0, top.size, size=(size, top.size))
        draws[start : start + size] = top[index].sum(axis=1) / bottom[index].sum(axis=1)
    tail = (1.0 - STUDY.margins.confidence_level) / 2.0
    return point, float(np.quantile(draws, 1.0 - tail))


def _reader(block: pd.DataFrame) -> str:
    """The candidate a cell's rows record: the rule's, or a control's."""
    if str(block["property"].iloc[0]) == study.CONTROL:
        return CONTROL_CANDIDATE[str(block["cell"].iloc[0]).split("__", 1)[1]]
    return "rule"


def comparator_of(block: pd.DataFrame) -> str:
    """The candidate a cell's rule is paired against and its width is read over."""
    construction = str(block["construction"].iloc[0])
    return "capped_t" if construction == "skew_t" else "identity_normal"


def cell_readings(block: pd.DataFrame) -> dict[str, Any]:
    """Every :data:`SUMMARY_COLUMNS` value of one cell's rows (stamped)."""
    name = _reader(block)
    control = str(block["property"].iloc[0]) == study.CONTROL
    requested = int(block["requested_replicates"].iloc[0])
    covered = block["covered"].to_numpy(dtype=int)
    withheld = withheld_triggers(block)
    shown = withheld == ""
    truth = _column(block, "truth")
    psi, se = _column(block, "estimate"), _column(block, "std_error")
    low, high = candidate_interval(block, name)
    out: dict[str, Any] = {"shown_replicates": int(shown.sum())}
    hits = int(covered[shown].sum())
    out["shown_coverage"] = hits / shown.sum() if shown.any() else math.nan
    out["shown_ci_lower"], out["shown_ci_upper"] = _interval(hits, int(shown.sum()))
    # The legs: the first LEG_REPLICATES replicates by index.
    legs = min(study.LEG_REPLICATES, requested)
    first = block["replicate"].to_numpy(dtype=int) < legs
    leg_hits = int(covered[first].sum())
    out["leg_replicates"] = legs
    out["leg_survivors"] = int(first.sum())
    out["leg_coverage"] = leg_hits / first.sum() if first.any() else math.nan
    out["leg_complete_ci_lower"] = _interval(leg_hits, int(first.sum()))[0]
    out["lower_leg_ci_lower"], out["lower_leg_ci_upper"] = _interval(leg_hits, legs)
    leg_shown = first & shown
    out["leg_shown"] = int(leg_shown.sum())
    leg_shown_hits = int(covered[leg_shown].sum())
    out["leg_shown_coverage"] = leg_shown_hits / leg_shown.sum() if leg_shown.any() else math.nan
    out["leg_shown_ci_upper"] = _interval(leg_shown_hits, int(leg_shown.sum()))[1]
    # The tail clause: all replicates, shown fits.
    with np.errstate(invalid="ignore"):
        below = int(np.sum(shown & (high < truth)))
        above = int(np.sum(shown & (low > truth)))
    out["below_count"], out["above_count"] = below, above
    out["below_ci_lower"] = tail_lower(below, int(shown.sum()))
    out["above_ci_lower"] = tail_lower(above, int(shown.sum()))
    out["tails_passed"] = tails_hold(below, above, int(shown.sum()))
    out["constant_cell_rate"] = float(np.mean(withheld == "constant_cell"))
    out["out_of_range_rate"] = float(np.mean(withheld == "out_of_range"))
    out["lower_miss_rate"] = below / len(block)
    out["upper_miss_rate"] = above / len(block)
    for level in study.REPORTED_LEVELS:
        at_low, at_high = candidate_interval(block, name, alpha=1.0 - level)
        out[f"coverage_{round(level * 100):02d}"] = float(np.mean(_covers(block, at_low, at_high)))
    nu = _column(block, "nu")
    nu = nu[np.isfinite(nu)]
    out["nu_median"] = float(np.median(nu)) if nu.size else math.nan
    accel = _column(block, "acceleration")
    accel = accel[np.isfinite(accel)]
    out["acceleration_median"] = float(np.median(accel)) if accel.size else math.nan
    error = psi - truth
    out["se_error_correlation"] = (
        float(np.corrcoef(se, error)[0, 1])
        if len(block) > 2 and np.std(se) > 0 and np.std(error) > 0
        else math.nan
    )
    spread = float(np.std(psi, ddof=1)) if len(block) > 1 else math.nan
    z = float(stats.norm.ppf(1.0 - study.ALPHA / 2.0))
    out["oracle_sd_coverage"] = float(np.mean(np.abs(error) <= z * spread))
    covers = {}
    for candidate in CANDIDATES:
        c_low, c_high = candidate_interval(block, candidate)
        applies = bool(np.isfinite(c_low).any() or np.isfinite(c_high).any())
        covers[candidate] = _covers(block, c_low, c_high)
        out[f"candidate_{candidate}"] = float(np.mean(covers[candidate])) if applies else math.nan
    comparator = comparator_of(block)
    out["comparator"] = comparator
    c_low, c_high = candidate_interval(block, comparator)
    rule_low, rule_high = candidate_interval(block, "rule")
    point, upper = width_ratio_bound(
        np.where(shown, rule_high - rule_low, np.nan),
        np.where(shown, c_high - c_low, np.nan),
        seed=stream_seed(STUDY, "width_ratio", str(block["cell"].iloc[0])),
    )
    out["width_ratio"], out["width_ratio_upper"] = point, upper
    if control:
        out["gain_b"] = out["gain_c"] = out["gain_lower"] = math.nan
        return out
    rule = covered.astype(bool)
    other = covers[comparator]
    b, c = int(np.sum(rule & ~other)), int(np.sum(~rule & other))
    out["gain_b"], out["gain_c"], out["gain_lower"] = b, c, gain_lower(b, c)
    return out


def leg_passes(row: pd.Series) -> bool:
    """Both declared legs on the first 2,000: lower leg over the requested, upper over shown."""
    low, high = STUDY.margins.calibration_coverage
    return bool(row["lower_leg_ci_lower"] >= low and row["leg_shown_ci_upper"] <= high)


def _expected_withheld(cell: str, n: int) -> float:
    source = cell.split("__", 1)[0]
    if source not in study.CELL_BY_NAME:
        return math.nan
    group, estimand = study.GROUP_OF_CELL[source], study.CELL_BY_NAME[source].estimand
    if group.law in study.HELDOUT_LAWS:
        return heldout.constant_cell(group.law, estimand)
    return laws.constant_cell_share(group.law, n, estimand)


def witness_rows(rows: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    """The deliberate-mutation witnesses: a source cell's fits under a mutated interval.

    ``passed`` is true when the mutation fails the heavy-side tail clause, as it must for the
    instrument to discriminate (plan 9.6).
    """
    out = []
    for name, source, mutation in study.WITNESSES:
        block = rows.loc[(rows["cell"] == source) & (rows["property"] == study.OWNER)]
        row: dict[str, Any] = dict.fromkeys(columns, math.nan)
        row.update(
            {
                "property": study.WITNESS,
                "cell": name,
                "role": DIAGNOSTIC,
                "n": int(block["n"].iloc[0]) if len(block) else math.nan,
                "replicates": len(block),
                "failed_replicates": 0,
                "passed": False,
                "property_passed": None,
            }
        )
        if not block.empty:
            low, high = candidate_interval(block, mutation)
            truth = _column(block, "truth")
            shown = withheld_triggers(block) == ""
            with np.errstate(invalid="ignore"):
                below = int(np.sum(shown & (high < truth)))
                above = int(np.sum(shown & (low > truth)))
            hits = int(np.sum(_covers(block, low, high)))
            row.update(
                {
                    "truth": float(truth[0]),
                    "coverage": hits / len(block),
                    "shown_replicates": int(shown.sum()),
                    "below_count": below,
                    "above_count": above,
                    "below_ci_lower": tail_lower(below, int(shown.sum())),
                    "above_ci_lower": tail_lower(above, int(shown.sum())),
                    "tails_passed": tails_hold(below, above, int(shown.sum())),
                    "passed": not tails_hold(below, above, int(shown.sum())),
                }
            )
        out.append(row)
    return out


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the split reading of plan 9.5, the gains and the witness rows."""
    margins = STUDY.margins
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=SUMMARY_COLUMNS, rate_labels=()
    )
    summary["comparator"] = summary["comparator"].astype(object)
    summary["tails_passed"] = summary["tails_passed"].astype(object)
    low, high = margins.calibration_coverage
    for index, line in summary.iterrows():
        family, cell = str(line["property"]), str(line["cell"])
        block = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        for column, value in cell_readings(block).items():
            summary.at[index, column] = value
        summary.at[index, "expected_withheld"] = _expected_withheld(cell, int(line["n"]))
        source = study.GROUP_OF_CELL.get(cell.split("__", 1)[0])
        if source is not None and source.law == "h2":
            requested = int(block["requested_replicates"].iloc[0])
            summary.at[index, "expected_failed"] = laws.h2_empty_cell(int(line["n"])) * requested
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ratio_intervals(
                block,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, "se_ratio", family, cell),
            )["se_ratio"]
        summary.at[index, "se_ratio_ci_lower"] = ratio.low
        summary.at[index, "se_ratio_ci_upper"] = ratio.high
        current = summary.loc[index]
        if line["role"] == DIAGNOSTIC:
            # No accounting corrects a reported or control row: it reads the declared legs.
            legs = leg_passes(current)
        else:
            # The complete case; the failure accounting turns it into the worst case.
            legs = bool(
                current["leg_complete_ci_lower"] >= low and current["leg_shown_ci_upper"] <= high
            )
        passed = legs and bool(current["tails_passed"])
        if cell in study.PAIRED_GAINS:
            passed = passed and float(current["gain_lower"]) > 0.5
        summary.at[index, "passed"] = passed
    witnesses = pd.DataFrame(witness_rows(rows, summary.columns))
    summary = pd.concat([summary, witnesses], ignore_index=True)
    return finish(summary.reset_index(drop=True), rates)


# ------------------------------------------------------------------------------ falsifiers


def _row(summary: pd.DataFrame, family: str, cell: str) -> pd.Series | None:
    found = summary.loc[(summary["property"] == family) & (summary["cell"] == cell)]
    return None if found.empty else found.iloc[0]


def lower_leg_fails(row: pd.Series) -> bool:
    """The declared lower leg on the first 2,000, withheld and lost fits as misses."""
    return bool(row["lower_leg_ci_lower"] < STUDY.margins.calibration_coverage[0])


def any_leg_fails(row: pd.Series) -> bool:
    """Either declared leg on the first 2,000 fails."""
    return not leg_passes(row)


def tails_fail(row: pd.Series) -> bool:
    """The tail clause fails on either side."""
    return not bool(row["tails_passed"])


#: The declared falsifiers of plan 9.6 that read legs and tails: name, target class, cells,
#: and whether any leg (``"any"``) or the lower leg alone (``"lower"``) counts.
LEG_FALSIFIERS = (
    ("mover", "bounded differences", ("d1_ate", "d2_ate"), "lower"),
    ("skew", "continuous targets", ("c1_ey1", "c1_ate", "c2_ey1"), "lower"),
    ("means", "bounded means (PR 2 rule)", ("d1_ey1", "d2_ey0"), "any"),
)


def mover_against_identity(rows: pd.DataFrame) -> tuple[int, int, float]:
    """``b``, ``c`` and the exact conditional 99.5% one-sided lower bound of ``c / (b + c)``.

    On ``d1_ate`` and ``d2_ate`` pooled, over all fits: ``b`` the fits MOVER covers and
    identity normal misses, ``c`` the reverse (plan 9.6).
    """
    block = rows.loc[(rows["property"] == study.HELDOUT) & rows["cell"].isin(("d1_ate", "d2_ate"))]
    if block.empty:
        return 0, 0, math.nan
    mover = block["covered"].to_numpy(dtype=int).astype(bool)
    identity = _covers(block, *candidate_interval(block, "identity_normal"))
    b, c = int(np.sum(mover & ~identity)), int(np.sum(~mover & identity))
    return b, c, gain_lower(c, b)


def falsifiers(summary: pd.DataFrame, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each declared falsifier's reading (plan 9.6); a missing cell leaves it unfired.

    Parameters
    ----------
    summary : pandas.DataFrame
        :func:`summarize_properties` of the rows.
    rows : pandas.DataFrame or None
        The property rows, for the paired comparison of MOVER with identity normal.  Without
        them that row is read from the summary's ``gain_b`` and ``gain_c`` of the two cells.

    Returns
    -------
    pandas.DataFrame
        One row per falsifier and cell.
    """
    out = []
    for name, target, cells, leg in LEG_FALSIFIERS:
        for cell in cells:
            row = _row(summary, study.HELDOUT, cell)
            if row is None:
                out.append(
                    {"falsifier": name, "target_class": target, "cell": cell, "reading": "",
                     "value": math.nan, "threshold": math.nan, "fired": False}
                )  # fmt: skip
                continue
            legs = lower_leg_fails(row) if leg == "lower" else any_leg_fails(row)
            tails = tails_fail(row)
            out.append(
                {
                    "falsifier": name,
                    "target_class": target,
                    "cell": cell,
                    "reading": f"{leg} leg failed: {legs}; tail clause failed: {tails}",
                    "value": float(row["lower_leg_ci_lower"]),
                    "threshold": STUDY.margins.calibration_coverage[0],
                    "fired": bool(legs or tails),
                }
            )
    rows_present = [_row(summary, study.HELDOUT, cell) for cell in ("d1_ate", "d2_ate")]
    if rows is not None:
        b, c, bound = mover_against_identity(rows)
    elif all(row is not None for row in rows_present):
        b = int(sum(int(row["gain_b"]) for row in rows_present if row is not None))
        c = int(sum(int(row["gain_c"]) for row in rows_present if row is not None))
        bound = gain_lower(c, b)
    else:
        b, c, bound = 0, 0, math.nan
    out.append(
        {
            "falsifier": "mover against identity",
            "target_class": "bounded differences",
            "cell": "d1_ate+d2_ate",
            "reading": f"b={b}, c={c}; lower bound of c/(b+c)",
            "value": bound,
            "threshold": 0.5,
            "fired": bool(math.isfinite(bound) and bound > 0.5),
        }
    )
    for cell in study.WIDTH_CELLS:
        row = _row(summary, study.HELDOUT, cell)
        value = math.nan if row is None else float(row["width_ratio_upper"])
        out.append(
            {
                "falsifier": "harm (width)",
                "target_class": "continuous targets",
                "cell": cell,
                "reading": "99% bootstrap upper bound of the mean width ratio over the capped t",
                "value": value,
                "threshold": study.WIDTH_THRESHOLD,
                "fired": bool(math.isfinite(value) and value > study.WIDTH_THRESHOLD),
            }
        )
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------ the rules


def _legs_lost(ctx: pf.Context, row: pd.Series) -> tuple[int, int, int]:
    """Hits, survivors and lost fits among the first 2,000 replicates of the row's cell."""
    block = ctx.cell(str(row["property"]), str(row["cell"]))
    legs = int(row["leg_replicates"])
    first = block["replicate"].to_numpy(dtype=int) < legs
    hits = int(block["covered"].to_numpy(dtype=int)[first].sum())
    return hits, int(first.sum()), legs - int(first.sum())


def _lower_leg() -> pf.Clause:
    """The lower leg on the first 2,000: each lost fit there may be a miss or a cover."""

    def judged(lower: float, ctx: pf.Context) -> pf.Judged:
        floor = ctx.margins.calibration_coverage[0]
        return pf.Judged(lower - floor, bool(lower >= floor))

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        return judged(float(row["leg_complete_ci_lower"]), ctx)

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        hits, survivors, lost = _legs_lost(ctx, row)
        return [
            (f"j={j} covered", judged(_interval(hits + j, survivors + lost)[0], ctx))
            for j in range(lost + 1)
        ]

    return pf.Clause(
        "coverage lower leg, first 2,000",
        "S2",
        pf.roles("positive"),
        read,
        candidates,
        imputation=pf.SUPPORT,
    )


def _upper_leg() -> pf.Clause:
    """The shown-only upper leg on the first 2,000: lost fits are absent by declaration."""

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        high = ctx.margins.calibration_coverage[1]
        value = float(row["leg_shown_ci_upper"])
        return pf.Judged(high - value, bool(value <= high))

    return pf.Clause(
        "coverage upper leg, first 2,000, shown only",
        "S2",
        pf.roles("positive"),
        read,
        lambda ctx, row: [("lost fits absent", read(ctx, row))],
        imputation=pf.SUPPORT,
    )


def _tail(side: str) -> pf.Clause:
    """The tail clause on one side: each lost fit may be a one-sided miss on that side."""
    column = f"{side}_count"

    def judged(count: int, shown: int) -> pf.Judged:
        bound = tail_lower(count, shown)
        return pf.Judged(study.TAIL_THRESHOLD - bound, bool(bound <= study.TAIL_THRESHOLD))

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        return judged(int(row[column]), int(row["shown_replicates"]))

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        count, shown, k = int(row[column]), int(row["shown_replicates"]), ctx.k(row)
        return [(f"j={j} {side}", judged(count + j, shown + j)) for j in range(k + 1)]

    return pf.Clause(
        f"tail clause, {side}", "S2", pf.roles("positive"), read, candidates, imputation=pf.SUPPORT
    )


def _paired_gain() -> pf.Clause:
    """The paired gain: each lost fit may be one the comparator alone covers."""

    def judged(b: int, c: int) -> pf.Judged:
        bound = gain_lower(b, c)
        return pf.Judged(bound - 0.5, bool(bound > 0.5))

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        return judged(int(row["gain_b"]), int(row["gain_c"]))

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        b, c = int(row["gain_b"]), int(row["gain_c"])
        return [(f"c+{j}", judged(b, c + j)) for j in range(ctx.k(row) + 1)]

    return pf.Clause(
        "paired gain",
        "S2",
        pf.cells(*study.PAIRED_GAINS),
        read,
        candidates,
        imputation=pf.SUPPORT,
    )


#: The failure accounting's rule of the gated families: the split legs, both tails, the gains.
PROPERTY_RULES = (
    pf.FamilyRule(
        families=frozenset(study.COVERAGE_FAMILIES),
        shared="asymmetric_wald_properties.summarize_properties",
        clauses=(_lower_leg(), _upper_leg(), _tail("below"), _tail("above"), _paired_gain()),
    ),
)

__all__ = [
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "falsifiers",
    "generate_property_artifacts",
    "generate_property_rows",
    "summarize_properties",
]
