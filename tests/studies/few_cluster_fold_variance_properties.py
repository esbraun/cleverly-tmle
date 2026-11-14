r"""The ``fold_variance_calibration`` family of ``few-cluster-fold-variance``.

Each key draws one sample of a cell of :data:`tests.studies.few_cluster_fold_variance.CELLS`,
fits it twice, and writes one row per arm
(:func:`tests.studies.few_cluster_fold_variance.draw_arms`). Beside the shared replicate
columns a row records ``withheld`` (the estimate withheld its interval, so the arm reads a
miss) and ``reference_df``. A key whose fit raises is lost in every arm of its cell.

Cells are read by **training clusters** :math:`J(V-1)/V` (2, 4, 6, 8, 16 and 24), the clusters
each fold's nuisances are fitted on, with the clusters per validation fold as a second axis
(plan X54 rev 3.2, section 2.4 item 1). The variance statistic is the **variance ratio** over
the shown fits of a cell,

.. math::

    \widehat{\mathrm{VR}} = \overline{\hat\sigma^2}\,/\,\widehat{\mathrm{Var}}(\hat\psi),

the mean squared SE over the ``ddof=1`` variance of the estimates, with a 99% percentile
bootstrap interval that resamples replicates jointly (:func:`variance_ratio_interval`).

**The positive rule** (:func:`positive_rule`), on each of the two gated configurations
(``t_reference``, the propensity estimated, and ``known_propensity``, declared known):

* ``shown coverage in band``: the exact 99% interval of coverage over the shown fits inside
  ``margins.calibration_coverage`` (0.92, 0.98);
* ``all-draw coverage floor`` (G1): the exact 99% lower end of coverage over every requested
  draw, withheld and lost fits as misses, at least :data:`G1_FLOOR` (0.90);
* ``variance ratio lower end``: the variance ratio's 99% lower end at least :math:`0.93^2 =
  0.8649`. The upper side is reported, not gated: it measures the mass of outlier SEs (plan
  2.4 item 4), and the coverage band's upper leg still gates a variance too large on typical
  draws.

**The controls**:

* ``deflated_control`` and ``known_deflated_control``, each positive configuration's variance
  times :data:`~tests.studies.few_cluster_fold_variance.DEFLATION`: the control passes when its
  rows **fail** the positive rule (``deflated variance fails the positive rule``). With the
  positive rule reduced to the variance ratio's lower end, the control asks the rule, not one of
  its legs, to see an underestimated variance;
* ``iid_t_control``: the median over the shown draws of the paired ratio of the IID SE to the
  cluster SE of the same fit, whose distribution-free exact 99% upper end
  (:func:`median_interval`) must lie below :data:`IID_MEDIAN_CEILING` (0.95). An inert cluster
  aggregation gives a ratio of exactly 1 on every draw. The ratio of means of Fix-4's clause 1 is
  reported (``paired_se_ratio``).

Every other row is ``diagnostic``. Every row publishes the training and validation clusters,
the shown and withheld counts, the shown and all-draw coverages with their exact intervals, the
variance ratio with its interval, the 5%-trimmed variance ratio, the largest SE in SDs of the
estimate, the median SE over that SD, the shown SE ratio with its interval, :math:`k = J - V`
and :math:`c_k = E\sqrt{\chi^2_k/k}`, and the over-coverage reading. :func:`training_floor`
reads each configuration's training-cluster floor off a summary (plan 5.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_fold_variance as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import Interval, clopper_pearson, percentile_interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    bootstrap_draw_blocks,
    paired_se_ratio_interval,
    se_ratio_interval,
)
from tests.studies.evidence.property_failures import (
    SUPPORT,
    Clause,
    Context,
    FamilyRule,
    Judged,
    above_judge,
    band_judge,
    cells,
    floor_judge,
    se_ratio_sets,
)
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAMILY: Final[str] = study.FAMILY

#: The variance-ratio band: the calibration SE-ratio band squared, :math:`[0.8649, 1.1449]`.
VARIANCE_BAND: Final[tuple[float, float]] = (
    STUDY.margins.calibration_se_ratio[0] ** 2,
    STUDY.margins.calibration_se_ratio[1] ** 2,
)
#: The two-sided band of the shown-fit coverage interval.
COVERAGE_BAND: Final[tuple[float, float]] = STUDY.margins.calibration_coverage
#: The ceiling of the IID control's median paired SE ratio's 99% upper end (plan 5.3, rev 3.2).
IID_MEDIAN_CEILING: Final[float] = 0.95
#: G1: the floor of the exact 99% lower end of all-draw coverage.
G1_FLOOR: Final[float] = 0.90
#: The share of the largest squared SEs the trimmed variance ratio leaves out (reported).
TRIM: Final[float] = 0.05
#: The over-coverage reading's threshold.
OVER_COVERAGE: Final[float] = STUDY.margins.over_coverage_ceiling

#: The columns a row carries beside :data:`REPLICATE_COLUMNS`.
EXTRA_COLUMNS: Final[tuple[str, ...]] = ("withheld", "reference_df")

#: The readings :func:`summarize_properties` adds to every row.
READING_COLUMNS: Final[tuple[str, ...]] = (
    "training_clusters",
    "validation_clusters",
    "shown_draws",
    "withheld_draws",
    "shown_coverage",
    "shown_ci_lower",
    "shown_ci_upper",
    "shown_above_099",
    "all_draw_coverage",
    "all_draw_ci_lower",
    "all_draw_ci_upper",
    "variance_ratio",
    "variance_ratio_ci_lower",
    "variance_ratio_ci_upper",
    "variance_ratio_trimmed",
    "largest_se_sd",
    "median_se_ratio",
    "shown_se_ratio",
    "shown_se_ratio_ci_lower",
    "shown_se_ratio_ci_upper",
    "reference_k",
    "jensen_factor",
    "paired_se_ratio",
    "paired_se_ratio_ci_lower",
    "paired_se_ratio_ci_upper",
    "paired_median_ratio",
    "paired_median_ci_lower",
    "paired_median_ci_upper",
    "coverage_band_passed",
    "g1_passed",
    "variance_lower_passed",
    "positive_rule_passed",
)


def _cell(cell: str, arm: str) -> str:
    return study.cell_name(cell, arm)


POSITIVE_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(cell.name, arm) for cell in study.CELLS for arm in study.POSITIVE_ARMS
)
IID_CELLS: Final[tuple[str, ...]] = tuple(_cell(cell.name, "iid_t_control") for cell in study.CELLS)
DEFLATED_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(cell.name, arm) for cell in study.CELLS for arm in study.DEFLATED_ARMS.values()
)
POOLED_CELLS: Final[tuple[str, ...]] = tuple(
    _cell(name, "pooled_centered") for name in sorted(study.POOLED_CENTERED_GATED)
)

#: The positive and IID-control arms of each cell, the pairs the IID control reads.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    FAMILY: tuple(
        (_cell(cell.name, "t_reference"), _cell(cell.name, "iid_t_control")) for cell in study.CELLS
    )
}


def _split(name: str) -> tuple[str, str]:
    """``(cell prefix, arm)`` of a published cell name."""
    prefix, arm = name.rsplit("__", 1)
    return prefix, arm


def _role(name: str) -> str:
    return study.arm_role(*_split(name))


# ------------------------------------------------------------------------------- statistics


def variance_ratio(estimates: np.ndarray, std_errors: np.ndarray) -> float:
    """``mean(se**2) / var(estimate, ddof=1)``; ``nan`` below two draws or at zero spread."""
    estimates = np.asarray(estimates, dtype=float)
    if estimates.size < 2:
        return float("nan")
    spread = float(np.var(estimates, ddof=1))
    if spread <= 0.0:
        return float("nan")
    return float(np.mean(np.square(np.asarray(std_errors, dtype=float))) / spread)


def variance_ratio_interval(
    estimates: np.ndarray,
    std_errors: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> Interval:
    """The percentile bootstrap interval of :func:`variance_ratio`.

    Each bootstrap draw resamples whole replicates, so a draw keeps each fit's estimate with
    its own SE, and the numerator and the denominator read the same resampled replicates.
    """
    values = np.column_stack(
        [np.asarray(estimates, dtype=float), np.asarray(std_errors, dtype=float)]
    )
    if len(values) < 2:
        return Interval(float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ratios = np.concatenate(
        [
            np.mean(np.square(block[:, :, 1]), axis=1) / np.var(block[:, :, 0], axis=1, ddof=1)
            for block in bootstrap_draw_blocks(values, replicates=replicates, rng=rng)
        ]
    )
    return percentile_interval(ratios, confidence_level=confidence_level)


def variance_seed(record: Any, cell: str) -> int:
    """The verdict's own resampling seed of one cell's variance ratio."""
    return int(stream_seed(record, FAMILY, cell, "variance_ratio"))


def median_interval(
    values: np.ndarray, *, confidence_level: float, high: int = 0, low: int = 0
) -> Interval:
    """The distribution-free exact interval of a median, from order statistics.

    With :math:`n` values sorted, the interval is :math:`[x_{(j)}, x_{(n-j+1)}]` with :math:`j`
    the ``(1 - level) / 2`` quantile of :math:`\\mathrm{Bin}(n, 1/2)`, so it covers the median
    with probability at least ``confidence_level`` for any continuous law. ``high`` and ``low``
    add that many values above and below every given one (the failure accounting's fills).
    """
    ordered = np.sort(np.asarray(values, dtype=float))
    ordered = np.concatenate([np.full(low, -np.inf), ordered, np.full(high, np.inf)])
    n = ordered.size
    if n == 0:
        return Interval(float("nan"), float("nan"))
    j = int(stats.binom.ppf((1.0 - confidence_level) / 2.0, n, 0.5))
    if j < 1:
        return Interval(float("-inf"), float("inf"))
    return Interval(float(ordered[j - 1]), float(ordered[n - j]))


def trimmed_variance_ratio(estimates: np.ndarray, std_errors: np.ndarray) -> float:
    """:func:`variance_ratio` with the largest :data:`TRIM` share of squared SEs left out."""
    squared = np.sort(np.square(np.asarray(std_errors, dtype=float)))
    kept = squared[: int(np.floor((1.0 - TRIM) * squared.size))]
    estimates = np.asarray(estimates, dtype=float)
    if kept.size == 0 or estimates.size < 2:
        return float("nan")
    spread = float(np.var(estimates, ddof=1))
    return float(np.mean(kept) / spread) if spread > 0 else float("nan")


def paired_ratios(iid: pd.DataFrame, positive: pd.DataFrame) -> np.ndarray:
    """IID SE over cluster SE on each replicate both rows hold and show."""
    left = _shown(iid).set_index("replicate")["std_error"]
    right = _shown(positive).set_index("replicate")["std_error"]
    joined = pd.concat([left.rename("iid"), right.rename("cluster")], axis=1, join="inner")
    return (joined["iid"] / joined["cluster"]).to_numpy(dtype=float)


# ------------------------------------------------------------------------------------- rows


def _seed(cell: str, replicate: int) -> int:
    return int(stream_seed(STUDY, "property_sample", FAMILY, cell, replicate))


def _draw(key: tuple[str, int, int]) -> pd.DataFrame:
    name, replicate, _ = key
    cell = study.CELL_BY_NAME[name]
    return study.draw(cell.law, cell.clusters, np.random.default_rng(_seed(name, replicate)))


def requested(cell: str, budget: int | None = None) -> int:
    """A cell's requested replications, capped at ``budget`` for a pre-run check."""
    declared = study.PROPERTY_REPLICATES[cell]
    return declared if budget is None else min(budget, declared)


def key_rows(
    name: str, replicate: int, entry: Mapping[str, Mapping[str, Any]], truth: float, wanted: int
) -> list[dict[str, Any]]:
    """The seven rows of one key, off :func:`~tests.studies.few_cluster_fold_variance.draw_arms`."""
    cell = study.CELL_BY_NAME[name]
    rows = []
    for arm in study.ARMS:
        published = _cell(name, arm)
        reading = entry[arm]
        rows.append(
            {
                "property": FAMILY,
                "cell": published,
                "role": _role(published),
                "replicate": replicate,
                "n": cell.n,
                "requested_replicates": wanted,
                "failed_replicates": 0,
                "truth": float(truth),
                "estimate": float(reading["psi"]),
                "std_error": float(reading["std_error"]),
                "covered": int(reading["covered"]),
                "rejected": int(reading["rejected"]),
                "withheld": int(bool(reading["withheld"])),
                "reference_df": int(reading["reference_df"]),
            }
        )
    return rows


def _fit_key(key: tuple[str, int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    name, replicate, wanted = key
    cell = study.CELL_BY_NAME[name]
    truth = study.law_truth(cell.law)[study.TARGETS[cell.law]]
    return key_rows(name, replicate, study.draw_arms(frame, cell, truth), truth, wanted)


def _cells_of(key: tuple[str, int, int]) -> list[str]:
    """Every arm of the key's cell reads its one fit."""
    return [f"{FAMILY}/{_cell(key[0], arm)}" for arm in study.ARMS]


def _cell_of(key: tuple[str, int, int]) -> str:
    return f"{FAMILY}/{_cell(key[0], 't_reference')}"


def _replicate_of(key: tuple[str, int, int]) -> int:
    return key[1]


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its role, size and requested replicates."""
    out = []
    for cell in study.CELLS:
        for arm in study.ARMS:
            name = _cell(cell.name, arm)
            out.append(
                PropertyCell(
                    property=FAMILY,
                    cell=name,
                    dgp=cell,
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=cell.n,
                    replicates=study.PROPERTY_REPLICATES[cell.name],
                    seed=_seed(cell.name, 0),
                    role=_role(name),
                    estimand=study.TARGETS[cell.law],
                )
            )
    return tuple(out)


#: The synthetic cell whose rows withhold at :data:`SYNTHETIC_WITHHELD`, so that G1 can decide a
#: verdict the shown-coverage band does not: with nothing withheld or lost, a shown lower end of
#: at least 0.92 implies an all-draw lower end of at least 0.90.
SYNTHETIC_WITHHELD_CELL: Final[str] = _cell("equal10__v5_j20", "t_reference")
SYNTHETIC_WITHHELD: Final[float] = 0.07
#: The deflated controls are centred on the truth in the synthetic rows, as a calibrated
#: variance would be, so that the positive rule passes on them and the control reads red.
SYNTHETIC_CENTRES: Final[dict[str, float]] = {f"{FAMILY}/{name}": 0.0 for name in DEFLATED_CELLS}


def synthetic_columns(cell: Any, rng: np.random.Generator) -> dict[str, Any]:
    """The extra columns of a synthetic row for the property differential."""
    withheld = 0
    if cell.cell == SYNTHETIC_WITHHELD_CELL:
        withheld = int(rng.uniform() < SYNTHETIC_WITHHELD)
    return {"withheld": withheld, "reference_df": int(cell.dgp.df)}


#: The share of a cell's requested replications that the accounting lets a run lose. The plan
#: asked for 0.10; the harness caps a bound at 0.05
#: (:data:`~tests.studies.evidence.property_failures.MAXIMUM_BOUND`). The pilot lost no fit.
FAILURE_BOUND = 0.05

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pre-run check."""
    keys = [
        (cell.name, replicate, requested(cell.name, budget))
        for cell in study.CELLS
        for replicate in range(requested(cell.name, budget))
    ]
    outcomes = fit_properties(
        keys,
        _fit_key,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *EXTRA_COLUMNS]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------------- readings


def _shown(group: pd.DataFrame) -> pd.DataFrame:
    return group.loc[group["withheld"].to_numpy(dtype=float) < 0.5]


def fold_variance_readings(summary: pd.DataFrame, rows: pd.DataFrame, record: Any) -> None:
    """Fill :data:`READING_COLUMNS` on every row of the family."""
    margins = record.margins
    level = margins.confidence_level
    family = rows.loc[rows["property"] == FAMILY]
    groups = dict(tuple(family.groupby("cell", sort=False)))
    for index in summary.index[summary["property"] == FAMILY]:
        name = str(summary.loc[index, "cell"])
        prefix, _ = _split(name)
        cell = study.CELL_BY_NAME[prefix]
        group = groups[name]
        shown = _shown(group)
        hits, trials = int(shown["covered"].sum()), len(shown)
        requested = int(group["requested_replicates"].iloc[0])
        values: dict[str, float] = {
            "training_clusters": float(cell.training),
            "validation_clusters": float(cell.per_fold),
            "shown_draws": float(trials),
            "withheld_draws": float(len(group) - trials),
            "reference_k": float(cell.df),
            "jensen_factor": study.jensen_factor(cell.df),
        }
        if trials:
            exact = clopper_pearson(hits, trials, confidence_level=level)
            values.update(
                shown_coverage=hits / trials,
                shown_ci_lower=exact.low,
                shown_ci_upper=exact.high,
                shown_above_099=float(exact.low > OVER_COVERAGE),
            )
        # G1: the shown hits over every requested draw, so a withheld or lost fit is a miss.
        every = clopper_pearson(hits, requested, confidence_level=level)
        values.update(
            all_draw_coverage=hits / requested,
            all_draw_ci_lower=every.low,
            all_draw_ci_upper=every.high,
        )
        estimates = shown["estimate"].to_numpy(dtype=float)
        std_errors = shown["std_error"].to_numpy(dtype=float)
        values["variance_ratio"] = variance_ratio(estimates, std_errors)
        values["variance_ratio_trimmed"] = trimmed_variance_ratio(estimates, std_errors)
        ratio = variance_ratio_interval(
            estimates,
            std_errors,
            replicates=margins.bootstrap_replicates,
            confidence_level=level,
            seed=variance_seed(record, name),
        )
        values["variance_ratio_ci_lower"], values["variance_ratio_ci_upper"] = ratio.low, ratio.high
        if trials >= 2:
            spread = float(np.std(estimates, ddof=1))
            if spread > 0:
                values["shown_se_ratio"] = float(np.mean(std_errors) / spread)
                values["largest_se_sd"] = float(np.max(std_errors) / spread)
                values["median_se_ratio"] = float(np.median(std_errors) / spread)
            se_ratio = se_ratio_interval(
                shown,
                replicates=margins.bootstrap_replicates,
                confidence_level=level,
                seed=stream_seed(record, FAMILY, name, "se_ratio"),
                truth_varies=False,
            )
            values["shown_se_ratio_ci_lower"] = se_ratio.low
            values["shown_se_ratio_ci_upper"] = se_ratio.high
        if name in IID_CELLS:
            positive = groups[_cell(prefix, "t_reference")]
            paired = paired_se_ratio_interval(
                group,
                positive,
                replicates=margins.bootstrap_replicates,
                confidence_level=level,
                seed=stream_seed(record, FAMILY, name),
            )
            values["paired_se_ratio"] = paired.ratio
            values["paired_se_ratio_ci_lower"] = paired.interval.low
            values["paired_se_ratio_ci_upper"] = paired.interval.high
            ratios = paired_ratios(group, positive)
            median = median_interval(ratios, confidence_level=level)
            values["paired_median_ratio"] = float(np.median(ratios)) if ratios.size else np.nan
            values["paired_median_ci_lower"] = median.low
            values["paired_median_ci_upper"] = median.high
        for column, value in values.items():
            summary.loc[index, column] = value
        if name in POSITIVE_CELLS or name in DEFLATED_CELLS:
            for column, value in gate_readings(summary.loc[index]).items():
                summary.loc[index, column] = float(value)
            summary.loc[index, "positive_rule_passed"] = float(positive_rule(summary.loc[index]))


def gate_readings(row: pd.Series) -> dict[str, bool]:
    """Each gate of the positive rule on one summary row (plan 5.3)."""
    return {
        "coverage_band_passed": bool(
            COVERAGE_BAND[0] <= row["shown_ci_lower"] and row["shown_ci_upper"] <= COVERAGE_BAND[1]
        ),
        "g1_passed": bool(row["all_draw_ci_lower"] >= G1_FLOOR),
        "variance_lower_passed": bool(row["variance_ratio_ci_lower"] >= VARIANCE_BAND[0]),
    }


def positive_rule(row: pd.Series) -> bool:
    """The three gates of a positive cell, read off its summary row (plan 5.3)."""
    return all(gate_readings(row).values())


def fold_variance_verdicts(summary: pd.DataFrame) -> None:
    """The declared verdicts of the family, read off its summary columns."""
    family = summary["property"] == FAMILY
    for index in summary.index[family]:
        row = summary.loc[index]
        name = str(row["cell"])
        if name in POSITIVE_CELLS:
            passed = positive_rule(row)
        elif name in IID_CELLS:
            passed = bool(row["paired_median_ci_upper"] < IID_MEDIAN_CEILING)
        elif name in DEFLATED_CELLS:
            passed = not positive_rule(row)
        elif name in POOLED_CELLS:
            passed = bool(row["variance_ratio_ci_lower"] > VARIANCE_BAND[1])
        else:
            passed = True
        summary.loc[index, "passed"] = passed
        summary.loc[index, "property_passed"] = passed


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the family's readings and its declared verdicts."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=READING_COLUMNS, rate_labels=()
    )
    fold_variance_readings(summary, rows, STUDY)
    fold_variance_verdicts(summary)
    return finish(summary, rates)


def _certified_counts(passed: pd.Series, arm: str) -> set[int]:
    """The training counts at which every positive cell of ``arm`` passes."""
    certified = dict.fromkeys(study.TRAINING_LADDER, True)
    for cell in study.CELLS:
        certified[cell.training] = certified[cell.training] and bool(passed[_cell(cell.name, arm)])
    return {count for count, ok in certified.items() if ok}


def training_floor(summary: pd.DataFrame, arm: str) -> int | None:
    """A configuration's training-cluster floor :math:`T^*` (plan 5.3), off a summary.

    The smallest count of :data:`~tests.studies.few_cluster_fold_variance.TRAINING_LADDER` from
    which every positive cell of the configuration, on both laws, passes. ``None`` when no count
    qualifies, including the largest. Whether the configuration may ship a floor at all is
    :func:`declared_route`'s question.
    """
    passed = summary.set_index("cell")["passed"].astype(bool)
    certified = _certified_counts(passed, arm)
    floor = None
    for count in sorted(study.TRAINING_LADDER, reverse=True):
        if count not in certified:
            break
        floor = count
    return floor


#: The outcomes of :func:`declared_route` (plan 5.8, rev 3.5).
CALIBRATED = "calibrated"
TRAINING_FLOOR = "training floor"
NO_FLOOR = "no floor (finite-sample)"
DEFECT_SUSPECT = "defect suspect"
UNLICENSED = "unlicensed"


@dataclass(frozen=True)
class Route:
    """The declared consequence of one configuration's verdicts (plan 5.8).

    Parameters
    ----------
    arm : str
        The positive arm, one of :data:`~tests.studies.few_cluster_fold_variance.POSITIVE_ARMS`.
    outcome : str
        :data:`CALIBRATED` (every cell passes: no floor ships), :data:`TRAINING_FLOOR`
        (``MINIMUM_FOLD_TRAINING_CLUSTERS[mechanism] = floor`` ships), :data:`NO_FLOOR` (a red at
        24 training clusters that a recorded diagnosis finds no defect: no floor ships, and the
        record states no certified training count), :data:`DEFECT_SUSPECT` (diagnose and fix
        before anything ships) or :data:`UNLICENSED` (a must-fail control did not fail, or the
        IID control saw no cluster effect, in some cell: nothing ships from the configuration,
        and the study is redesigned under a fresh declaration).
    floor : int or None
        :func:`training_floor`.
    minimum : int or None
        The value of ``MINIMUM_FOLD_TRAINING_CLUSTERS[mechanism]`` the outcome ships, or ``None``.
    owners : dict
        Each red positive cell of the configuration with the owner and reading that hold it,
        ``(RED_OWNER, reading)``. ``None`` marks a red no owner holds, which is why the outcome
        is a defect suspect.
    non_monotone : tuple of str
        The red positive cells at a training count above a count at which every positive cell
        of the configuration passes: the signature of the finite-sample reading.
    reasons : tuple of str
        Why the outcome is what it is, one entry per failing condition.
    """

    arm: str
    outcome: str
    floor: int | None
    minimum: int | None
    owners: dict[str, tuple[str, str] | None]
    non_monotone: tuple[str, ...]
    reasons: tuple[str, ...]


def declared_route(
    summary: pd.DataFrame,
    arm: str,
    diagnoses: Mapping[str, str] | None = None,
) -> Route:
    """Map a summary to the route plan 5.8 declares for one configuration.

    1. Licence: in every cell the configuration's deflated control must fail the positive rule
       and the IID control must pass. Otherwise the instrument could not see the failure it
       targets, and the configuration is :data:`UNLICENSED`, whatever its positive cells read.
    2. Floor: :func:`training_floor`.
    3. A red positive cell whose variance-ratio lower end passes is held by
       :data:`~tests.studies.few_cluster_fold_variance.RED_OWNER` when it sits below the floor,
       or when there is no floor and ``diagnoses`` records a no-defect diagnosis for ``arm``. Its
       reading is :data:`~tests.studies.few_cluster_fold_variance.FINITE_SAMPLE_READING` at or
       above the configuration's
       :data:`~tests.studies.few_cluster_fold_variance.PREDICTED_FLOOR`, or on the non-monotone
       signature, and :data:`~tests.studies.few_cluster_fold_variance.TRAINING_FLOOR_READING`
       otherwise. A red variance-ratio lower end is an underestimated variance, which no owner
       holds: a :data:`DEFECT_SUSPECT`.
    4. No floor is a :data:`DEFECT_SUSPECT`, unless ``diagnoses`` records the diagnosis for
       ``arm`` and every red is held: then :data:`NO_FLOOR`.
    5. Otherwise :data:`CALIBRATED` at a floor of 2, else :data:`TRAINING_FLOOR`.

    ``diagnoses`` defaults to
    :data:`~tests.studies.few_cluster_fold_variance.NO_DEFECT_DIAGNOSES`, which is empty before
    the run.
    """
    if diagnoses is None:
        diagnoses = study.NO_DEFECT_DIAGNOSES
    rows = summary.loc[summary["property"] == FAMILY].set_index("cell")
    passed = rows["passed"].astype(bool)
    reasons: list[str] = []
    control = study.DEFLATED_ARMS[arm]
    for cell in study.CELLS:
        if not passed[_cell(cell.name, control)]:
            reasons.append(f"{_cell(cell.name, control)} passes the positive rule")
        if not passed[_cell(cell.name, "iid_t_control")]:
            reasons.append(f"{_cell(cell.name, 'iid_t_control')} sees no cluster effect")
    floor = training_floor(summary, arm)
    certified = _certified_counts(passed, arm)
    diagnosed = floor is None and arm in diagnoses
    owners: dict[str, tuple[str, str] | None] = {}
    non_monotone: list[str] = []
    for cell in study.CELLS:
        name = _cell(cell.name, arm)
        if passed[name]:
            continue
        if any(count < cell.training for count in certified):
            non_monotone.append(name)
        below = floor is not None and cell.training < floor
        if not ((below or diagnosed) and bool(rows.loc[name, "variance_lower_passed"])):
            owners[name] = None
            continue
        finite = cell.training >= study.PREDICTED_FLOOR[arm] or name in non_monotone
        reading = study.FINITE_SAMPLE_READING if finite else study.TRAINING_FLOOR_READING
        owners[name] = (study.RED_OWNER, reading)
    route = partial(Route, arm, floor=floor, owners=owners, non_monotone=tuple(non_monotone))
    if reasons:
        return route(UNLICENSED, minimum=None, reasons=tuple(reasons))
    if floor is None and not diagnosed:
        reasons.append("no training count from which every cell passes")
    reasons.extend(
        f"{name} reads a variance-ratio lower end below 0.8649"
        for name, owner in owners.items()
        if owner is None and (floor is not None or diagnosed)
    )
    if reasons:
        return route(DEFECT_SUSPECT, minimum=None, reasons=tuple(reasons))
    if floor is None:
        return route(NO_FLOOR, minimum=None, reasons=(diagnoses[arm],))
    if floor == study.TRAINING_LADDER[0]:
        return route(CALIBRATED, minimum=None, reasons=())
    return route(TRAINING_FLOOR, minimum=floor, reasons=())


# ------------------------------------------------------------------------------ declared rules


def _variance_candidates(ctx: Context, row: pd.Series, judge: Any) -> list[tuple[str, Judged]]:
    """The S5 candidate set of a variance-ratio interval over the shown fits.

    A lost draw is either withheld, so it leaves the shown fits unchanged, or shown at an
    extreme of the survivors: the SE at the survivor maximum with the estimates at their mean,
    and the SE at the survivor minimum with the estimates at each vertex split
    (:func:`~tests.studies.evidence.property_failures.se_ratio_sets`). The mean squared SE is
    monotone in its fills and the variance convex in its own, so the set holds both extremes of
    the point ratio, as it does for the SE ratio.
    """
    shown = _shown(ctx.cell(FAMILY, str(row["cell"])))
    margins = ctx.margins

    def interval(frame: pd.DataFrame) -> Interval:
        return variance_ratio_interval(
            frame["estimate"].to_numpy(dtype=float),
            frame["std_error"].to_numpy(dtype=float),
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=variance_seed(ctx.record, str(row["cell"])),
        )

    out = [("every lost draw withheld", judge(ctx, row, interval(shown)))]
    for label, frame in se_ratio_sets(shown, ctx.k(row), truth_varies=False):
        out.append((label, judge(ctx, row, interval(frame))))
    return out


def _variance_clause(name: str, judge: Any, applies: Any) -> Clause:
    """A variance-ratio interval clause over the shown fits, with the S5 candidate set."""

    def read(ctx: Context, row: pd.Series) -> Judged:
        iv = Interval(float(row["variance_ratio_ci_lower"]), float(row["variance_ratio_ci_upper"]))
        return judge(ctx, row, iv)

    return Clause(name, "S5", applies, read, lambda c, r: _variance_candidates(c, r, judge))


_VARIANCE_FLOOR = floor_judge(lambda _: VARIANCE_BAND[0])
_COVERAGE_BAND = band_judge(lambda _: COVERAGE_BAND)
_G1 = floor_judge(lambda _: G1_FLOOR)


def _coverage_read(ctx: Context, row: pd.Series) -> Judged:
    iv = Interval(float(row["shown_ci_lower"]), float(row["shown_ci_upper"]))
    return _COVERAGE_BAND(ctx, row, iv)


def _coverage_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """A lost draw is withheld or shown; a shown one is covered or not: every split."""
    trials = int(row["shown_draws"])
    hits = round(float(row["shown_coverage"]) * trials) if trials else 0
    level = ctx.margins.confidence_level
    out = []
    k = ctx.k(row)
    for shown in range(k + 1):
        for covered in range(shown + 1):
            if trials + shown == 0:
                continue
            iv = clopper_pearson(hits + covered, trials + shown, confidence_level=level)
            out.append((f"{shown} shown, {covered} covered", _COVERAGE_BAND(ctx, row, iv)))
    return out


def _g1_read(ctx: Context, row: pd.Series) -> Judged:
    iv = Interval(float(row["all_draw_ci_lower"]), float(row["all_draw_ci_upper"]))
    return _G1(ctx, row, iv)


def _g1_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """G1 counts every requested draw; a lost one is covered or not: every count."""
    requested = int(ctx.cell(FAMILY, str(row["cell"]))["requested_replicates"].iloc[0])
    hits = round(float(row["all_draw_coverage"]) * requested)
    level = ctx.margins.confidence_level
    return [
        (
            f"j={j} covered",
            _G1(ctx, row, clopper_pearson(hits + j, requested, confidence_level=level)),
        )
        for j in range(ctx.k(row) + 1)
    ]


def _fails_rule(gates: Sequence[Judged]) -> Judged:
    """A control's reading: the rule fails when some gate fails, by its largest shortfall."""
    return Judged(max(-gate.slack for gate in gates), any(not gate.passed for gate in gates))


def _deflated_read(ctx: Context, row: pd.Series) -> Judged:
    return _fails_rule(
        [
            _coverage_read(ctx, row),
            _g1_read(ctx, row),
            Judged(
                float(row["variance_ratio_ci_lower"]) - VARIANCE_BAND[0],
                bool(float(row["variance_ratio_ci_lower"]) >= VARIANCE_BAND[0]),
            ),
        ]
    )


def _deflated_candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
    """Each gate at its own most passing fill of the lost draws.

    A joint fill gives every gate a reading at least as failing as its own most passing fill,
    so a rule that fails under these readings fails under every joint fill.
    """

    def best(candidates: list[tuple[str, Judged]]) -> Judged:
        return max((judged for _, judged in candidates), key=lambda judged: judged.slack)

    gates = [
        best(_coverage_candidates(ctx, row)),
        best(_g1_candidates(ctx, row)),
        best(_variance_candidates(ctx, row, _VARIANCE_FLOOR)),
    ]
    return [("each gate at its most passing fill", _fails_rule(gates))]


def _median_clause() -> Clause:
    """The IID control: the median paired IID/cluster SE ratio's upper end below 0.95.

    A lost key costs both arms of the pair, so a lost draw adds one ratio of unknown value: the
    order-statistic interval's upper end is monotone in each added value, so the candidates add
    ``j`` ratios above every survivor and the rest below.
    """

    def judged(iv: Interval) -> Judged:
        return Judged(IID_MEDIAN_CEILING - iv.high, bool(iv.high < IID_MEDIAN_CEILING))

    def read(ctx: Context, row: pd.Series) -> Judged:
        return judged(
            Interval(float(row["paired_median_ci_lower"]), float(row["paired_median_ci_upper"]))
        )

    def positive(row: pd.Series) -> str:
        return _cell(_split(str(row["cell"]))[0], "t_reference")

    def candidates(ctx: Context, row: pd.Series) -> list[tuple[str, Judged]]:
        ratios = paired_ratios(ctx.cell(FAMILY, str(row["cell"])), ctx.cell(FAMILY, positive(row)))
        k = ctx.k(row)
        level = ctx.margins.confidence_level
        return [
            (
                f"{j} above, {k - j} below",
                judged(median_interval(ratios, confidence_level=level, high=j, low=k - j)),
            )
            for j in range(k + 1)
        ]

    def reads(ctx: Context, row: pd.Series) -> list[tuple[str, str]]:
        return [(FAMILY, str(row["cell"])), (FAMILY, positive(row))]

    return Clause(
        "median paired iid/cluster se ratio under its ceiling",
        "S9m",
        cells(*IID_CELLS),
        read,
        candidates,
        imputation=SUPPORT,
        reads=reads,
    )


#: The failure accounting's declared rules: the verdicts of :func:`fold_variance_verdicts`.
PROPERTY_RULES = (
    FamilyRule(
        families=frozenset({FAMILY}),
        shared="summarize_properties",
        clauses=(
            Clause(
                "shown coverage in band",
                "S2",
                cells(*POSITIVE_CELLS),
                _coverage_read,
                _coverage_candidates,
                imputation=SUPPORT,
            ),
            Clause(
                "all-draw coverage floor",
                "S2",
                cells(*POSITIVE_CELLS),
                _g1_read,
                _g1_candidates,
                imputation=SUPPORT,
            ),
            _variance_clause("variance ratio lower end", _VARIANCE_FLOOR, cells(*POSITIVE_CELLS)),
            _median_clause(),
            Clause(
                "deflated variance fails the positive rule",
                "S5",
                cells(*DEFLATED_CELLS),
                _deflated_read,
                _deflated_candidates,
            ),
            _variance_clause(
                "pooled-centred variance ratio above band",
                above_judge(lambda _: VARIANCE_BAND[1]),
                cells(*POOLED_CELLS),
            ),
        ),
    ),
)


__all__: Sequence[str] = (
    "PROPERTY_RULES",
    "Route",
    "declared_cells",
    "declared_route",
    "generate_property_rows",
    "summarize_properties",
    "training_floor",
    "variance_ratio",
    "variance_ratio_interval",
)
