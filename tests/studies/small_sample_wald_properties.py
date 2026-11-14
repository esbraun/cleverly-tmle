r"""The property rows of ``small-sample-wald``, their summary, verdicts and falsifiers.

Each group of :data:`tests.studies.small_sample_wald.GROUPS` is one draw per replicate, keyed
``(group, replicate)`` and drawn from ``stream_seed(STUDY, "property_sample", group,
replicate)``.  Every fit of the draw writes one row per cell it fills
(:func:`~tests.studies.small_sample_wald.rule_columns`): the shared columns with the plug-in
standard error, the rule's interval, and the small-sample columns, so every candidate interval
of plan 2.8 is recomputed from the rows.  ``covered`` is 1 when the rule's interval is shown
and covers the truth: a withheld interval is a miss.  ``rejected`` is ``1 - covered``, the rule
interval's test of the true value.

Each two-sided coverage cell also writes one ``control`` row per control of
:func:`~tests.studies.small_sample_wald.controls_of`, the same fit read under the control's
interval, with the same withheld trigger.  Controls are reported (:data:`DIAGNOSTIC`).

**Legs.**  A failed fit (an H2 sample with an empty design cell, or any fit error) is a
recorded property failure.  The *lower leg* is the shared ``coverage`` interval, withheld fits
as misses; the failure accounting's worst case reads every lost fit as a miss too, which is the
declared lower leg over all ``R`` fits (plan 5.1).  The *upper leg* reads the shown fits only
(``shown_ci_upper``); a lost fit is absent from it by declaration, like a withheld one.
``lower_leg_ci_lower`` and ``lower_leg_ci_upper`` publish the declared lower leg over all
requested fits directly, and every reading the accounting does not correct reads it
(:func:`leg_passes`): the falsifiers, the controls and the reported rows.

**Verdicts** (:func:`summarize_properties`): a positive two-sided coverage cell passes when its
lower leg's 99% lower end is at least 0.92 and its shown-only 99% upper end at most 0.98; a
paired-gain cell also needs the exact conditional 99.5% lower bound of ``b / (b + c)`` above
1/2; a positive ``non_h1`` cell needs its lower leg's lower end at least 0.92, and the control
its coverage's upper end below 0.90.  Every other row is reported, with ``passed`` holding its
own reading: both legs for a coverage row, and for the leverage rows also the SE-ratio
interval containing 1 (plan 5.2.1, review 3 B-2).  ``constant_cell_witness`` is a summary row:
among the constant-cell fits of ``boundary_mean_n500_ey1``, the plug-in interval on the
range-logit scale (the point itself when the estimate sits on a bound) must cover with 99%
upper end below 0.50.

``falsifiers.csv`` (:func:`falsifiers`) evaluates the four declared falsifiers of plan 5.3 on
the held-out cells and their controls.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tests.parallel import STUDY_JOBS
from tests.studies import small_sample_laws as laws
from tests.studies import small_sample_wald as study
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

#: The candidate intervals of plan 2.8, each read off the rows.
CANDIDATES = ("identity_normal", "capped_t", "kish_t", "range_logit_normal", "logit_t")

#: The candidate each control reads.
CONTROL_CANDIDATE = {
    "normal_reference_control": "identity_normal",
    "kish_control": "kish_t",
    "logit_t_control": "logit_t",
    "capped_t_binary_control": "capped_t",
}

#: The columns the summary publishes beside the shared ones.
SUMMARY_COLUMNS = (
    "shown_replicates",
    "shown_coverage",
    "shown_ci_lower",
    "shown_ci_upper",
    "lower_leg_coverage",
    "lower_leg_ci_lower",
    "lower_leg_ci_upper",
    "constant_cell_rate",
    "out_of_range_rate",
    "expected_constant_cell",
    "lower_miss_rate",
    "upper_miss_rate",
    *(f"coverage_{round(level * 100):02d}" for level in study.REPORTED_LEVELS),
    "nu_median",
    "nu_p10",
    "nu_p90",
    "se_error_correlation",
    "oracle_sd_coverage",
    *(f"candidate_{name}" for name in CANDIDATES),
    "gain_b",
    "gain_c",
    "gain_lower",
    "expected_failed",
    "paired_se_ratio_difference",
    "paired_se_ratio_difference_ci_lower",
    "paired_se_ratio_difference_ci_upper",
)

#: The leverage pairs whose SE ratios are differenced on their shared draws (plan 5.2.1).
LEVERAGE_PAIRS = (
    ("leverage_crossfit_ey1", "leverage_in_sample_ey1"),
    ("leverage_crossfit_ate", "leverage_in_sample_ate"),
)


# ------------------------------------------------------------------------------ intervals


def candidate_interval(
    frame: pd.DataFrame, name: str, *, alpha: float = study.ALPHA
) -> tuple[np.ndarray, np.ndarray]:
    """One candidate interval of every row of ``frame``; ``nan`` where it does not apply.

    ``identity_normal``, ``capped_t`` (``max(nu, 5)``), ``kish_t`` (``max(kish, 5)``) are on
    the identity scale; ``range_logit_normal`` and ``logit_t`` (``max(nu, 5)``) are on the
    range-logit scale and apply to rows with bounds.  A missing ``nu`` reads as the normal.
    """
    psi = frame["estimate"].to_numpy(dtype=float)
    se = frame["std_error"].to_numpy(dtype=float)
    nu = np.nan_to_num(frame["nu"].to_numpy(dtype=float), nan=math.inf)
    kish = np.nan_to_num(frame["kish"].to_numpy(dtype=float), nan=math.inf)
    capped = np.maximum(nu, study.DF_CAP)
    df = {
        "identity_normal": np.full(len(frame), math.inf),
        "capped_t": capped,
        "kish_t": np.maximum(kish, study.DF_CAP),
        "range_logit_normal": np.full(len(frame), math.inf),
        "logit_t": capped,
    }[name]
    logit = name in ("range_logit_normal", "logit_t")
    return study.vector_interval(
        psi,
        se,
        low=frame["bound_low"].to_numpy(dtype=float),
        high=frame["bound_high"].to_numpy(dtype=float),
        df=df,
        logit=logit,
        alpha=alpha,
    )


def rule_vectors(frame: pd.DataFrame, *, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """The rule's interval of every row at level ``1 - alpha``, from the rows."""
    logit = np.isfinite(frame["bound_low"].to_numpy(dtype=float))
    gaussian = frame["outcome_family"].to_numpy() == "gaussian"
    out_low = np.full(len(frame), np.nan)
    out_high = np.full(len(frame), np.nan)
    for name, mask in (
        ("range_logit_normal", logit),
        ("capped_t", ~logit & gaussian),
        ("identity_normal", ~logit & ~gaussian),
    ):
        low, high = candidate_interval(frame, name, alpha=alpha)
        out_low[mask], out_high[mask] = low[mask], high[mask]
    return out_low, out_high


def withheld_triggers(frame: pd.DataFrame) -> np.ndarray:
    """Each row's withheld trigger, ``""`` for a shown interval.

    A fit writes ``""`` for a shown interval, and the committed Parquet file reads it back as
    null, so the summary recomputed from the committed rows normalizes both to ``""``.
    """
    return frame["withheld"].fillna("").astype(str).to_numpy()


def _covers(frame: pd.DataFrame, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Shown and covering; a withheld or undefined interval is a miss."""
    truth = frame["truth"].to_numpy(dtype=float)
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
    fits = study.fit_group(group, frame)
    rows: list[dict[str, Any]] = []
    for cell in group.cells:
        estimate = fits[cell.fit][cell.estimand]
        reference = float(truth[cell.estimand])
        extra = study.rule_columns(frame, cell.estimand, estimate, group.family)
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
        The group labels to fit; ``None`` is every group.

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
    return falsifiers(summarize_properties(stamped(rows)))


# ------------------------------------------------------------------------------ the summary


def _interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    iv = clopper_pearson(successes, trials, confidence_level=STUDY.margins.confidence_level)
    return iv.low, iv.high


def gain_lower(b: int, c: int, level: float = study.PAIRED_GAIN_LEVEL) -> float:
    """The exact conditional one-sided lower bound of ``b / (b + c)`` at ``level``."""
    return float(stats.beta.ppf(1.0 - level, b, c + 1)) if b > 0 else 0.0


def _reader(block: pd.DataFrame) -> Any:
    """The interval a cell's rows record at any level: the rule's, or a control's candidate."""
    if str(block["property"].iloc[0]) == study.CONTROL:
        name = CONTROL_CANDIDATE[str(block["cell"].iloc[0]).split("__", 1)[1]]
        return lambda alpha: candidate_interval(block, name, alpha=alpha)
    return lambda alpha: rule_vectors(block, alpha=alpha)


def cell_readings(block: pd.DataFrame) -> dict[str, Any]:
    """Every :data:`SUMMARY_COLUMNS` value of one cell's rows (stamped).

    The miss rates and the other levels read the interval the rows' ``covered`` reads: the
    rule's, or on a control row its control's candidate.  The paired gain compares the rule
    with the identity normal interval, so a control row leaves it empty.
    """
    interval = _reader(block)
    control = str(block["property"].iloc[0]) == study.CONTROL
    requested = int(block["requested_replicates"].iloc[0])
    covered = block["covered"].to_numpy(dtype=int)
    withheld = withheld_triggers(block)
    shown = withheld == ""
    truth = block["truth"].to_numpy(dtype=float)
    psi = block["estimate"].to_numpy(dtype=float)
    se = block["std_error"].to_numpy(dtype=float)
    low, high = interval(study.ALPHA)
    out: dict[str, Any] = {"shown_replicates": int(shown.sum())}
    hits = int(covered[shown].sum())
    out["shown_coverage"] = hits / shown.sum() if shown.any() else math.nan
    out["shown_ci_lower"], out["shown_ci_upper"] = _interval(hits, int(shown.sum()))
    out["lower_leg_coverage"] = covered.sum() / requested
    out["lower_leg_ci_lower"], out["lower_leg_ci_upper"] = _interval(int(covered.sum()), requested)
    out["constant_cell_rate"] = float(np.mean(withheld == "constant_cell"))
    out["out_of_range_rate"] = float(np.mean(withheld == "out_of_range"))
    with np.errstate(invalid="ignore"):
        out["lower_miss_rate"] = float(np.mean(shown & (truth < low)))
        out["upper_miss_rate"] = float(np.mean(shown & (truth > high)))
    for level in study.REPORTED_LEVELS:
        at_low, at_high = interval(1.0 - level)
        out[f"coverage_{round(level * 100):02d}"] = float(np.mean(_covers(block, at_low, at_high)))
    nu = block["nu"].to_numpy(dtype=float)
    nu = nu[np.isfinite(nu)]
    for name, q in (("nu_median", 0.5), ("nu_p10", 0.1), ("nu_p90", 0.9)):
        out[name] = float(np.quantile(nu, q)) if nu.size else math.nan
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
    for name in CANDIDATES:
        c_low, c_high = candidate_interval(block, name)
        applies = bool(np.isfinite(c_low).any() or np.isfinite(c_high).any())
        covers[name] = _covers(block, c_low, c_high)
        out[f"candidate_{name}"] = float(np.mean(covers[name])) if applies else math.nan
    if control:
        out["gain_b"] = out["gain_c"] = out["gain_lower"] = math.nan
        return out
    rule = covered.astype(bool)
    identity = covers["identity_normal"]
    b, c = int(np.sum(rule & ~identity)), int(np.sum(~rule & identity))
    out["gain_b"], out["gain_c"], out["gain_lower"] = b, c, gain_lower(b, c)
    return out


def leg_passes(row: pd.Series) -> bool:
    """Both declared legs: the lower leg over all requested fits, the upper over shown fits.

    The lower leg reads ``lower_leg_ci_lower``, withheld and failed fits as misses over every
    requested fit (plan 5.1, B-1).  A reading no accounting corrects (a falsifier, a control, a
    reported row) reads this, not the complete-case ``coverage_ci_lower``.
    """
    low, high = STUDY.margins.calibration_coverage
    return bool(row["lower_leg_ci_lower"] >= low and row["shown_ci_upper"] <= high)


def paired_ratio_difference(
    left: pd.DataFrame, right: pd.DataFrame, *, seed: int
) -> tuple[float, float, float]:
    """The difference of two cells' SE ratios on their shared replicates, with its interval.

    Each SE ratio is the mean reported SE over the SD of the estimates.  The bootstrap draws
    replicates jointly for both cells, :attr:`Margins.bootstrap_replicates` times, and the
    interval is the percentile one at the study's confidence level.
    """
    margins = STUDY.margins
    joined = left.merge(right, on="replicate", suffixes=("_l", "_r"))
    if len(joined) < 3:
        return math.nan, math.nan, math.nan

    def ratio(se: np.ndarray, psi: np.ndarray) -> np.ndarray:
        return se.mean(axis=-1) / psi.std(axis=-1, ddof=1)

    arrays = {
        key: joined[key].to_numpy(dtype=float)
        for key in ("std_error_l", "estimate_l", "std_error_r", "estimate_r")
    }
    point = float(
        ratio(arrays["std_error_l"], arrays["estimate_l"])
        - ratio(arrays["std_error_r"], arrays["estimate_r"])
    )
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(joined), size=(margins.bootstrap_replicates, len(joined)))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = ratio(arrays["std_error_l"][index], arrays["estimate_l"][index]) - ratio(
            arrays["std_error_r"][index], arrays["estimate_r"][index]
        )
    tail = (1.0 - margins.confidence_level) / 2.0
    low, high = np.nanquantile(draws, [tail, 1.0 - tail])
    return point, float(low), float(high)


def _law_of(cell: str) -> tuple[str, str] | None:
    source = cell.split("__", 1)[0]
    if source not in study.CELL_BY_NAME:
        return None
    return study.GROUP_OF_CELL[source].law, study.CELL_BY_NAME[source].estimand


def witness_row(rows: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """The ``constant_cell_witness`` summary row, from the source cell's constant-cell fits."""
    source = rows.loc[rows["cell"] == study.WITNESS_SOURCE]
    fired = source.loc[withheld_triggers(source) == "constant_cell"]
    row: dict[str, Any] = dict.fromkeys(columns, math.nan)
    row.update(
        {
            "property": study.WITNESS,
            "cell": "constant_cell_witness",
            "role": DIAGNOSTIC,
            "n": int(source["n"].iloc[0]) if len(source) else math.nan,
            "replicates": len(fired),
            "failed_replicates": 0,
            "passed": False,
            "property_passed": None,
        }
    )
    if fired.empty:
        return row
    low, high = candidate_interval(fired, "range_logit_normal")
    psi = fired["estimate"].to_numpy(dtype=float)
    low, high = np.where(np.isnan(low), psi, low), np.where(np.isnan(high), psi, high)
    truth = fired["truth"].to_numpy(dtype=float)
    hits = int(np.sum((low <= truth) & (truth <= high)))
    lower, upper = _interval(hits, len(fired))
    row.update(
        {
            "truth": float(truth[0]),
            "coverage": hits / len(fired),
            "coverage_ci_lower": lower,
            "coverage_ci_upper": upper,
            "passed": bool(upper < study.WITNESS_CEILING),
        }
    )
    return row


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the legs, the readings of 5.1, the verdicts and the witness row."""
    margins = STUDY.margins
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=SUMMARY_COLUMNS, rate_labels=()
    )
    low, high = margins.calibration_coverage
    for index, line in summary.iterrows():
        family, cell = str(line["property"]), str(line["cell"])
        block = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        for column, value in cell_readings(block).items():
            summary.at[index, column] = value
        source = _law_of(cell)
        if source is not None:
            summary.at[index, "expected_constant_cell"] = laws.constant_cell_share(
                source[0], int(line["n"]), source[1]
            )
            if source[0] == "h2":
                # B-1: H2's failed fits against their exact expectation.
                requested = int(block["requested_replicates"].iloc[0])
                summary.at[index, "expected_failed"] = (
                    laws.h2_empty_cell(int(line["n"])) * requested
                )
        with np.errstate(divide="ignore", invalid="ignore"):
            # A cell whose estimates do not vary (fits on a bound) has no finite ratio.
            ratio = ratio_intervals(
                block,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, "se_ratio", family, cell),
            )["se_ratio"]
        summary.at[index, "se_ratio_ci_lower"] = ratio.low
        summary.at[index, "se_ratio_ci_upper"] = ratio.high
        if line["role"] == DIAGNOSTIC:
            # No accounting corrects a reported row, so it reads the declared lower leg.
            legs = leg_passes(summary.loc[index])
        else:
            # The complete case; the failure accounting turns it into the worst case, which is
            # the declared lower leg (D13).
            legs = bool(
                line["coverage_ci_lower"] >= low and summary.at[index, "shown_ci_upper"] <= high
            )
        if family == study.NON_H1:
            passed = bool(
                line["coverage_ci_lower"] >= study.NON_H1_FLOOR
                if line["role"] == "positive"
                else line["coverage_ci_upper"] < study.NON_H1_CONTROL_CEILING
            )
        elif cell.startswith("leverage_"):
            passed = legs and ratio.low <= 1.0 <= ratio.high
        elif cell in study.PAIRED_GAIN_CELLS:
            passed = legs and summary.at[index, "gain_lower"] > 0.5
        else:
            passed = legs
        summary.at[index, "passed"] = passed
    for left, right in LEVERAGE_PAIRS:
        blocks = [rows.loc[rows["cell"] == name] for name in (left, right)]
        if any(block.empty for block in blocks):
            continue
        point, lower, upper = paired_ratio_difference(
            *blocks, seed=stream_seed(STUDY, "se_ratio_difference", left, right)
        )
        mask = summary["cell"].isin((left, right))
        summary.loc[mask, "paired_se_ratio_difference"] = point
        summary.loc[mask, "paired_se_ratio_difference_ci_lower"] = lower
        summary.loc[mask, "paired_se_ratio_difference_ci_upper"] = upper
    witness = pd.DataFrame([witness_row(rows, summary.columns)])
    summary = pd.concat([summary, witness], ignore_index=True)
    return finish(summary.reset_index(drop=True), rates)


# ------------------------------------------------------------------------------ falsifiers


def _row(summary: pd.DataFrame, family: str, cell: str) -> pd.Series | None:
    found = summary.loc[(summary["property"] == family) & (summary["cell"] == cell)]
    return None if found.empty else found.iloc[0]


def _lower_fails(row: pd.Series | None) -> bool | None:
    if row is None:
        return None
    return bool(row["lower_leg_ci_lower"] < STUDY.margins.calibration_coverage[0])


def _upper_fails(row: pd.Series | None) -> bool | None:
    if row is None:
        return None
    return bool(row["shown_ci_upper"] > STUDY.margins.calibration_coverage[1])


def _both_pass(row: pd.Series | None) -> bool | None:
    if row is None:
        return None
    return leg_passes(row)


#: The declared falsifiers of plan 5.3: name, target class, the cell, the leg that must fail,
#: and the control that must pass both legs on the same fits.
FALSIFIERS = (
    ("scale", "bounded means", "ho_sparse_treat_ey1", "lower", "capped_t_binary_control"),
    ("composition", "bounded means", "ho_sparse_treat_ey1", "lower", "logit_t_control"),
    ("composition", "bounded differences", "ho_sparse_treat_ate", "lower", "logit_t_control"),
    ("df", "continuous targets", "ho_hurdle_ey1", "lower", "kish_control"),
    ("df", "continuous targets", "ho_hurdle_ate", "lower", "kish_control"),
    ("t", "continuous targets", "ho_hurdle_ey1", "upper", "normal_reference_control"),
    ("t", "continuous targets", "ho_hurdle_ate", "upper", "normal_reference_control"),
    ("binary coefficient", "binary coefficients", "ho_logit_msm_a", "lower",
     "capped_t_binary_control"),
    ("binary coefficient", "binary coefficients", "ho_logit_msm_w", "lower",
     "capped_t_binary_control"),
)  # fmt: skip


def falsifiers(summary: pd.DataFrame) -> pd.DataFrame:
    """Each declared falsifier's reading: whether the cell's leg failed, the control passed.

    A falsifier fires when both hold.  A cell or control the summary lacks (a pre-run check
    that fitted no held-out law) leaves the row's readings empty and ``fired`` false.

    Parameters
    ----------
    summary : pandas.DataFrame
        :func:`summarize_properties` of the rows.

    Returns
    -------
    pandas.DataFrame
        One row per :data:`FALSIFIERS` entry.
    """
    out = []
    for name, target, cell, leg, control in FALSIFIERS:
        row = _row(summary, study.HELDOUT, cell)
        check = _row(summary, study.CONTROL, f"{cell}__{control}")
        failed = _lower_fails(row) if leg == "lower" else _upper_fails(row)
        passed = _both_pass(check)
        out.append(
            {
                "falsifier": name,
                "target_class": target,
                "cell": cell,
                "leg": leg,
                "control": control,
                "leg_failed": failed,
                "control_passed": passed,
                "fired": bool(failed and passed),
            }
        )
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------ the rules


def _upper_leg() -> pf.Clause:
    """The shown-only upper leg: lost fits are absent from it by declaration."""

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        high = ctx.margins.calibration_coverage[1]
        value = float(row["shown_ci_upper"])
        return pf.Judged(high - value, bool(value <= high))

    return pf.Clause(
        "coverage upper leg, shown only",
        "S2",
        pf.roles("positive"),
        read,
        lambda ctx, row: [("lost fits absent", read(ctx, row))],
        imputation=pf.SUPPORT,
    )


def _paired_gain() -> pf.Clause:
    """The paired gain: each lost fit may be one the identity normal interval alone covers."""

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
        pf.cells(*study.PAIRED_GAIN_CELLS),
        read,
        candidates,
        imputation=pf.SUPPORT,
    )


def _coverage_floor(ctx: pf.Context) -> float:
    return ctx.margins.calibration_coverage[0]


#: The failure accounting's rules: the two legs and the paired gain of the coverage families,
#: and the one-sided non-H1 cells.
PROPERTY_RULES = (
    pf.FamilyRule(
        families=frozenset(study.COVERAGE_FAMILIES),
        shared="small_sample_wald_properties.summarize_properties",
        clauses=(
            pf.proportion_clause(
                "coverage lower leg",
                statistic="S2",
                judge=pf.floor_judge(_coverage_floor),
                applies=pf.roles("positive"),
            ),
            _upper_leg(),
            _paired_gain(),
        ),
    ),
    pf.FamilyRule(
        families=frozenset({study.NON_H1}),
        shared="small_sample_wald_properties.summarize_properties",
        clauses=(
            pf.proportion_clause(
                "non-H1 lower end",
                statistic="S2",
                judge=pf.floor_judge(lambda ctx: study.NON_H1_FLOOR),
                applies=pf.roles("positive"),
            ),
            pf.proportion_clause(
                "non-H1 control upper end",
                statistic="S2",
                judge=pf.below_judge(lambda ctx: study.NON_H1_CONTROL_CEILING),
                applies=pf.roles("control"),
            ),
        ),
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
