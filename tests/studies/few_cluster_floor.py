r"""Shared pieces of the few-cluster floor studies (X54 parts (a), (b) and (c)).

``few-cluster-floor-tmle`` (:mod:`tests.studies.few_cluster_floor_tmle`) and
``few-cluster-floor-cross-fitted-ltmle`` (:mod:`tests.studies.few_cluster_floor_ltmle`)
measure the clustered interval of a fit family at every cluster count of a ladder below the
family's current floor, and compare three references on the same draws (part (c)). This module
holds what both read: the arms of one fit, the row each arm writes, the floor rule, the part (c)
statistics and decision, and the sizing functions the design test recomputes.

**Arms.** Each fit of a draw writes one row per arm, all with the fit's point estimate:

=========================  ==========================================================================
arm                        variance and reference
=========================  ==========================================================================
``t_reference``            the reported interval: the cluster IC variance on :math:`t_{J-2}`
                           (:math:`t_{J-V}` for a fold-evaluated fit); candidate (A), positive
``iid_t_control``          the same point with the IID row variance on the same quantile (control)
``normal_reference``       the cluster variance on the normal quantile (reported)
``t_j_minus_1_reference``  the cluster variance on :math:`t_{J-1}` (reported)
``t_j_minus_2_reference``  a fold-evaluated fit's variance on :math:`t_{J-2}` (reported)
``wang_inflated``          candidate (B): :math:`\frac{J}{J-p}\hat\sigma^2` on :math:`t_{J-p}`,
                           :math:`p = \max(2, d)`, recomputed from (A); undefined when
                           :math:`J \le p` (reported)
``cluster_jackknife``      candidate (C): the same fit refitted with
                           ``cluster_variance="jackknife"`` on :math:`t_{J-2}`, on the first
                           :data:`CANDIDATE_DRAWS` draws of a cell (reported)
=========================  ==========================================================================

A row records ``withheld`` (1 when the arm shows no interval: the fit withheld it, (B) is
undefined, or the jackknife refused or lost a refit) with its ``trigger``, so a withheld row is a
miss in every all-draw statistic and is left out of the shown-only ones.

**The floor** (plan 5.8). A count passes for a family when every positive configuration passes
G1 and G2 and its control passes, on every size law it runs on; the family floor is the smallest
ladder count from which every count passes (:func:`family_floor`).

**Part (c)** (plan 5.7). On the first :data:`CANDIDATE_DRAWS` draws of each cell, (B) and (C)
are judged against (A) by paired non-inferiority of all-draw coverage, per cell (99% lower end
above :data:`PER_CELL_MARGIN`) and pooled over the family's cells at or above the admissibility
count (above :data:`POOLED_MARGIN`); calibration is the shown-only exact interval inside
:data:`CALIBRATION_BAND`; length is the paired mean half-width ratio to (A) with a 99%
percentile-bootstrap interval. :func:`reference_decision` applies the declared decision.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln, gammaln

from cleverly.inference import influence_variance
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism
from tests.studies.clustered_few_cluster_tmle import ExactLongitudinalPropensity
from tests.studies.evidence.inference import Interval, clopper_pearson, percentile_interval
from tests.studies.evidence.properties import bootstrap_draw_blocks
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The family every positive and control row belongs to: the registered two-clause rule of
#: ``clustered-few-cluster-tmle`` (``property_verdicts._few_cluster_rule``) judges it.
FAMILY: Final[str] = "few_cluster_reference"
#: Every row's role in that family.
POSITIVE, CONTROL, DIAGNOSTIC = "positive", "control", "diagnostic"
ALPHA: Final[float] = 0.05
LEVEL: Final[float] = 0.99

#: The draws of each cell every part (c) statistic reads, for all three candidates (M3).
CANDIDATE_DRAWS: Final[int] = 2_000
#: Paired non-inferiority margins of (B) and (C) to (A): per cell and pooled (review 2 N2).
PER_CELL_MARGIN: Final[float] = -0.03
POOLED_MARGIN: Final[float] = -0.01
#: The shown-only exact coverage interval of a calibrated count.
CALIBRATION_BAND: Final[tuple[float, float]] = (0.92, 0.98)
#: How many more calibrated counts a candidate needs to lead on calibration alone.
CALIBRATION_LEAD: Final[int] = 2
#: G1's floor on the exact 99% lower end of all-draw coverage.
G1_FLOOR: Final[float] = 0.90
#: The all-draw pilot coverage from which a count is in the expected floor (M5).
EXPECTED_FLOOR_COVERAGE: Final[float] = 0.925
#: The family-wise plug-in pass probability the sizing must reach, its R steps and its cap.
FAMILY_PASS_TARGET: Final[float] = 0.95
R_STEP: Final[int] = 400
R_CAP: Final[int] = 6_000
#: The share of a cell's requested replications the accounting lets a run lose. The plan asks
#: for 0.10; the harness caps a bound at 0.05
#: (:data:`~tests.studies.evidence.property_failures.MAXIMUM_BOUND`). A lost fit is a miss in
#: G1 either way.
FAILURE_BOUND: Final[float] = 0.05

#: The positive arm, candidate (A).
REFERENCE_ARM: Final[str] = "t_reference"
#: Candidates (B) and (C).
WANG_ARM: Final[str] = "wang_inflated"
JACKKNIFE_ARM: Final[str] = "cluster_jackknife"
CANDIDATE_ARMS: Final[tuple[str, ...]] = (REFERENCE_ARM, WANG_ARM, JACKKNIFE_ARM)
#: The columns a row carries beside the shared replicate columns.
EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    "withheld",
    "trigger",
    "reference_df",
    "clusters",
    "half_width",
    "failed_refits",
)
#: The withheld triggers a row can record; ``""`` is a shown interval.
TRIGGERS: Final[tuple[str, ...]] = (
    "",
    "constant_cell",
    "constant_node",
    "out_of_range",
    "boundary_node",
    "wang_undefined",
    "jackknife_refused",
    "jackknife_refit_plugin",
    "other",
)


# --------------------------------------------------------------------------------- learners


class ParametricQuasiBinomialGLM(QuasiBinomialGLM):
    """The study GLM, declared parametric so that ``cluster_variance="jackknife"`` admits it.

    The R5 learner rule (:func:`cleverly.inference.cluster_jackknife.jackknife_learner`) reads
    the exact class or its ``cleverly_learner_family`` marker. The fit is the unpenalized
    quasibinomial IRLS of its parent, a finite-dimensional estimating equation.
    """

    cleverly_learner_family = "parametric"


class ParametricExactPropensity(ExactLongitudinalPropensity):
    """The informative law's exact propensity, declared parametric (it has no parameters)."""

    cleverly_learner_family = "parametric"


class ParametricKnownMechanism(KnownLongitudinalMechanism):
    """The end-of-study law's generating mechanism, declared parametric (no parameters)."""

    cleverly_learner_family = "parametric"


# ---------------------------------------------------------------------------------- quantiles


def t_critical(df: float) -> float:
    """The two-sided t quantile with ``df`` degrees of freedom; ``inf`` is the normal."""
    if not math.isfinite(df):
        return float(stats.norm.ppf(1.0 - ALPHA / 2.0))
    return float(stats.t.ppf(1.0 - ALPHA / 2.0, df))


NORMAL_CRITICAL: Final[float] = t_critical(math.inf)


def wang_parameters(clusters: int, covariates: int) -> tuple[float, int] | None:
    r"""(B)'s variance factor :math:`J/(J-p)` and df :math:`J-p`, :math:`p = \max(2, d)`.

    ``None`` when :math:`J \le p`: the construction has no reference there.
    """
    p = max(2, covariates)
    if clusters <= p:
        return None
    return clusters / (clusters - p), clusters - p


def trigger_of(estimate: Any) -> str:
    """The withheld trigger of an estimate: ``""`` when it shows its interval."""
    if bool(estimate.supplies_inference):
        return ""
    status = str(estimate.inference)
    for name, trigger in (
        ("constant_cell", "constant_cell"),
        ("constant_node", "constant_node"),
        ("out_of_range", "out_of_range"),
        ("boundary_node", "boundary_node"),
        ("jackknife_refit", "jackknife_refit_plugin"),
    ):
        if name in status:
            return trigger
    return "other"


# --------------------------------------------------------------------------------------- rows


@dataclasses.dataclass(frozen=True)
class Reading:
    """One arm of one fit: a point, a standard error, a quantile, and whether it is shown.

    Parameters
    ----------
    psi : float
        The point estimate.
    std_error : float
        The standard error; ``nan`` only when withheld.
    critical : float
        The two-sided quantile.
    reference_df : float
        The reference's degrees of freedom, ``inf`` for the normal.
    trigger : str
        One of :data:`TRIGGERS`; ``""`` when shown.
    failed_refits : int
        Failed jackknife refits (the (C) arm), else 0.
    """

    psi: float
    std_error: float
    critical: float
    reference_df: float
    trigger: str = ""
    failed_refits: int = 0

    @property
    def shown(self) -> bool:
        """Whether the arm shows an interval."""
        return self.trigger == ""


def reference_readings(
    estimate: Any,
    *,
    clusters: int,
    covariates: int,
    arms: Sequence[str],
) -> dict[str, Reading]:
    """Every arm in ``arms`` but (C), off one reported estimate.

    The positive arm reads the estimate's own reported standard error and ``reference_df``;
    the others keep its point and swap the variance or the quantile. An arm of a withheld
    estimate is withheld with the same trigger.
    """
    psi = float(estimate.psi)
    se = float(estimate.plugin_std_error)
    df = float(estimate.reference_df) if estimate.reference_df is not None else math.inf
    trigger = trigger_of(estimate)
    if not (math.isfinite(se) and se > 0.0):
        # A withheld estimate keeps its computed SE when it has one; otherwise the row is
        # withheld with no SE, as the row checks allow.
        if not trigger:  # pragma: no cover - a shown interval always has a positive SE
            raise AssertionError(f"a shown estimate reports the standard error {se}")
        se = float("nan")
    out: dict[str, Reading] = {}
    for arm in arms:
        if arm == REFERENCE_ARM:
            out[arm] = Reading(psi, se, t_critical(df), df, trigger)
        elif arm == "iid_t_control":
            iid = float(np.sqrt(influence_variance(estimate.influence_curve)))
            if not (math.isfinite(iid) and iid > 0.0):
                iid = float("nan")
            out[arm] = Reading(psi, iid, t_critical(df), df, trigger)
        elif arm == "normal_reference":
            out[arm] = Reading(psi, se, NORMAL_CRITICAL, math.inf, trigger)
        elif arm == "t_j_minus_1_reference":
            out[arm] = Reading(psi, se, t_critical(clusters - 1), clusters - 1, trigger)
        elif arm == "t_j_minus_2_reference":
            out[arm] = Reading(psi, se, t_critical(clusters - 2), clusters - 2, trigger)
        elif arm == WANG_ARM:
            parameters = wang_parameters(clusters, covariates)
            if parameters is None:
                out[arm] = Reading(psi, float("nan"), float("nan"), math.nan, "wang_undefined")
            else:
                factor, wang_df = parameters
                out[arm] = Reading(
                    psi, se * math.sqrt(factor), t_critical(wang_df), wang_df, trigger
                )
        elif arm == JACKKNIFE_ARM:
            continue
        else:  # pragma: no cover - a declared arm without a construction is a design defect
            raise KeyError(arm)
    return out


def jackknife_reading(estimate: Any, *, clusters: int) -> Reading:
    """(C) off an estimate of a ``cluster_variance="jackknife"`` fit, on :math:`t_{J-2}`.

    A refit failure or any other withheld status is a miss with its trigger.
    """
    record = estimate.jackknife
    if record is None:  # pragma: no cover - the study asked for the jackknife
        raise AssertionError("the (C) arm's estimate carries no jackknife record")
    failed = len(record.failed_refits)
    psi = float(estimate.psi)
    trigger = trigger_of(estimate)
    variance = float(record.jackknife_variance)
    if trigger or not (math.isfinite(variance) and variance > 0.0):
        return Reading(
            psi,
            float("nan"),
            float("nan"),
            clusters - 2,
            trigger or "jackknife_refit_plugin",
            failed,
        )
    df = float(estimate.reference_df)
    return Reading(psi, float(estimate.plugin_std_error), t_critical(df), df, "", failed)


def refused_jackknife(psi: float, clusters: int) -> Reading:
    """(C) on a draw where the jackknife refused before any refit: a miss."""
    return Reading(psi, float("nan"), float("nan"), clusters - 2, "jackknife_refused")


def arm_row(
    reading: Reading,
    *,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    clusters: int,
) -> dict[str, Any]:
    """The row one arm of one fit writes.

    A shown row covers when its interval holds ``truth``; a withheld row is a miss
    (``covered = 0``, ``rejected = 0``) with ``std_error`` ``nan`` only when the arm has none.
    """
    shown = reading.shown
    half = reading.critical * reading.std_error if shown else float("nan")
    low, high = reading.psi - half, reading.psi + half
    return {
        "property": FAMILY,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": float(truth),
        "estimate": reading.psi,
        "std_error": reading.std_error,
        "covered": int(shown and low <= truth <= high),
        "rejected": int(shown and not low <= 0.0 <= high),
        "withheld": int(not shown),
        "trigger": reading.trigger,
        "reference_df": float(reading.reference_df),
        "clusters": clusters,
        "half_width": half,
        "failed_refits": reading.failed_refits,
    }


#: The package-GLM columns an ``ltmle_in_sample`` positive row carries.
PACKAGE_COLUMNS: Final[tuple[str, ...]] = (
    "package_glm_raised",
    "package_glm_shown",
    "package_glm_covered",
    "package_glm_error",
)
#: The readings :func:`floor_readings` adds to every row of the family.
READING_COLUMNS: Final[tuple[str, ...]] = (
    "clusters",
    "shown_draws",
    "withheld_draws",
    "withheld_by_trigger",
    "shown_coverage",
    "shown_ci_lower",
    "shown_ci_upper",
    "shown_above_099",
    "all_draw_coverage",
    "all_draw_ci_lower",
    "all_draw_ci_upper",
    "shown_se_ratio",
    "package_glm_raised_share",
    "package_glm_raised_ci_lower",
    "package_glm_raised_ci_upper",
    "package_glm_coverage",
    "shown_coverage_package_raised",
    "shown_coverage_package_ran",
)


def parse(cell: str) -> tuple[str, str, int, str]:
    """``(fit, law, J, arm)`` of a published cell name."""
    fit, law, count, arm = cell.split("__")
    return fit, law, int(count.removeprefix("j")), arm


def floor_readings(summary: pd.DataFrame, rows: pd.DataFrame, record: Any) -> None:
    """Fill :data:`READING_COLUMNS` on every row of the family."""
    level = record.margins.confidence_level
    summary["withheld_by_trigger"] = summary["withheld_by_trigger"].astype(object)
    family = rows.loc[rows["property"] == FAMILY]
    groups = dict(tuple(family.groupby("cell", sort=False)))
    for index in summary.index[summary["property"] == FAMILY]:
        name = str(summary.loc[index, "cell"])
        _, _, clusters, arm = parse(name)
        group = groups[name]
        shown = group.loc[group["withheld"] == 0]
        hits, trials = int(shown["covered"].sum()), len(shown)
        wanted = int(group["requested_replicates"].iloc[0])
        triggers = group.loc[group["withheld"] == 1, "trigger"].value_counts().sort_index()
        values: dict[str, Any] = {
            "clusters": float(clusters),
            "shown_draws": float(trials),
            "withheld_draws": float(len(group) - trials),
            "withheld_by_trigger": ";".join(f"{k}={v}" for k, v in triggers.items()),
        }
        if trials:
            exact = clopper_pearson(hits, trials, confidence_level=level)
            values.update(
                shown_coverage=hits / trials,
                shown_ci_lower=exact.low,
                shown_ci_upper=exact.high,
                shown_above_099=float(exact.low > record.margins.over_coverage_ceiling),
            )
        every = clopper_pearson(hits, wanted, confidence_level=level)
        values.update(
            all_draw_coverage=hits / wanted,
            all_draw_ci_lower=every.low,
            all_draw_ci_upper=every.high,
        )
        if trials >= 2:
            spread = float(np.std(shown["estimate"].to_numpy(dtype=float), ddof=1))
            if spread > 0:
                values["shown_se_ratio"] = float(shown["std_error"].mean() / spread)
        if arm == REFERENCE_ARM and "package_glm_raised" in group:
            values.update(package_readings(group, level))
        for column, value in values.items():
            summary.loc[index, column] = value


def package_readings(group: pd.DataFrame, level: float) -> dict[str, float]:
    """The package-GLM readings of an ``ltmle_in_sample`` positive cell (M2, Q15)."""
    raised = pd.to_numeric(group["package_glm_raised"], errors="coerce")
    if raised.isna().all():
        return {}
    raised = raised.fillna(0).astype(int)
    count = int(raised.sum())
    exact = clopper_pearson(count, len(group), confidence_level=level)
    ran = group.loc[raised == 0]
    out = {
        "package_glm_raised_share": count / len(group),
        "package_glm_raised_ci_lower": exact.low,
        "package_glm_raised_ci_upper": exact.high,
        "package_glm_coverage": float(
            pd.to_numeric(ran["package_glm_covered"]).sum() / max(len(ran), 1)
        ),
    }
    for label, mask in (("raised", raised == 1), ("ran", raised == 0)):
        part = group.loc[mask & (group["withheld"] == 0)]
        out[f"shown_coverage_package_{label}"] = (
            float(part["covered"].mean()) if len(part) else float("nan")
        )
    return out


# ------------------------------------------------------------------------------------ floors


def family_floor(passes: Mapping[int, bool], *, anchor: int) -> int:
    """The family floor :math:`F_f`: the smallest count from which every ladder count passes.

    ``passes`` maps every ladder count to whether it passes (every positive configuration on
    every law, with its control). A failure at or above ``anchor`` keeps the floor at
    ``anchor``, the current floor (plan 5.8: a defect suspect, diagnosed first); counts above
    the anchor are part (c)'s and never raise a floor.
    """
    floor = anchor
    for count in sorted(passes, reverse=True):
        if not passes[count]:
            break
        if count <= anchor:
            floor = count
    return floor


def count_passes(
    verdicts: Mapping[tuple[str, str, int], bool], configurations: Iterable[str]
) -> dict[int, bool]:
    """Fold per-cell verdicts into per-count passes.

    ``verdicts`` maps ``(configuration, law, count)`` to whether that cell's positive and
    control both passed. A count passes when every configuration in ``configurations``
    passes on every law it was run on at that count (B4: both mechanism configurations of a
    point family).
    """
    wanted = set(configurations)
    counts: dict[int, bool] = {}
    seen: dict[int, set[str]] = {}
    for (configuration, _law, count), passed in verdicts.items():
        if configuration not in wanted:
            continue
        counts[count] = counts.get(count, True) and bool(passed)
        seen.setdefault(count, set()).add(configuration)
    for count, configs in seen.items():
        if configs != wanted:
            raise ValueError(f"count {count} lacks configurations {sorted(wanted - configs)}")
    return counts


# -------------------------------------------------------------------------- part (c) readings


@dataclasses.dataclass(frozen=True)
class PairedCounts:
    """The paired all-draw coverage of a candidate and (A) on the same draws.

    Parameters
    ----------
    draws : int
        The requested draws read (lost and withheld are misses in both arms).
    candidate_only : int
        Draws covered by the candidate and not by (A).
    reference_only : int
        Draws covered by (A) and not by the candidate.
    """

    draws: int
    candidate_only: int
    reference_only: int

    @property
    def difference(self) -> float:
        """The candidate's coverage minus (A)'s."""
        return (self.candidate_only - self.reference_only) / self.draws if self.draws else 0.0

    @property
    def variance(self) -> float:
        """The variance of :attr:`difference` (the paired multinomial form)."""
        if not self.draws:
            return 0.0
        b, c, n = self.candidate_only, self.reference_only, self.draws
        return max((b + c) - (b - c) ** 2 / n, 0.0) / n**2


def paired_counts(candidate: pd.DataFrame, reference: pd.DataFrame, *, draws: int) -> PairedCounts:
    """Discordant counts over replicates ``0 .. draws - 1``; a row absent there is a miss."""
    left = candidate.loc[candidate["replicate"] < draws].set_index("replicate")["covered"]
    right = reference.loc[reference["replicate"] < draws].set_index("replicate")["covered"]
    index = pd.RangeIndex(draws)
    a = left.reindex(index, fill_value=0).to_numpy(dtype=int)
    b = right.reindex(index, fill_value=0).to_numpy(dtype=int)
    return PairedCounts(draws, int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))


def difference_interval(pairs: Sequence[PairedCounts], *, level: float = LEVEL) -> Interval:
    """The normal interval of the draw-weighted mean coverage difference over cells.

    One cell is the per-cell clause; every cell of a family at or above its admissibility
    count is the pooled clause. Cells are independent, so variances add.
    """
    total = sum(pair.draws for pair in pairs)
    if not total:
        return Interval(float("nan"), float("nan"))
    weights = [pair.draws / total for pair in pairs]
    centre = sum(w * pair.difference for w, pair in zip(weights, pairs, strict=True))
    spread = math.sqrt(sum(w**2 * pair.variance for w, pair in zip(weights, pairs, strict=True)))
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    return Interval(centre - z * spread, centre + z * spread)


def calibrated(hits: int, shown: int, *, level: float = LEVEL) -> bool:
    """Whether the shown-only exact coverage interval lies inside :data:`CALIBRATION_BAND`."""
    if shown == 0:
        return False
    exact = clopper_pearson(hits, shown, confidence_level=level)
    return bool(CALIBRATION_BAND[0] <= exact.low and exact.high <= CALIBRATION_BAND[1])


def length_ratio(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    draws: int,
    replicates: int,
    seed: int,
    level: float = LEVEL,
) -> tuple[float, Interval]:
    """The paired mean half-width ratio to (A) on draws both show, with its bootstrap interval."""
    left = candidate.loc[(candidate["replicate"] < draws) & (candidate["withheld"] == 0)]
    right = reference.loc[(reference["replicate"] < draws) & (reference["withheld"] == 0)]
    joined = pd.concat(
        [
            left.set_index("replicate")["half_width"].rename("candidate"),
            right.set_index("replicate")["half_width"].rename("reference"),
        ],
        axis=1,
        join="inner",
    )
    if len(joined) < 2:
        return float("nan"), Interval(float("nan"), float("nan"))
    values = joined.to_numpy(dtype=float)
    point = float(values[:, 0].mean() / values[:, 1].mean())
    rng = np.random.default_rng(seed)
    ratios = np.concatenate(
        [
            block[:, :, 0].mean(axis=1) / block[:, :, 1].mean(axis=1)
            for block in bootstrap_draw_blocks(values, replicates=replicates, rng=rng)
        ]
    )
    return point, percentile_interval(ratios, confidence_level=level)


@dataclasses.dataclass(frozen=True)
class CandidateCell:
    """The part (c) reading of one candidate in one cell of one family.

    Parameters
    ----------
    candidate : str
        One of :data:`CANDIDATE_ARMS`.
    configuration : str
        The fit configuration (a mechanism configuration of a point family, or the family).
    law : str
        The size law.
    clusters : int
        The cluster count.
    pair : PairedCounts
        The paired counts against (A); (A) against itself for (A).
    hits : int
        Shown draws that cover, over the first :data:`CANDIDATE_DRAWS`.
    shown : int
        Shown draws over the first :data:`CANDIDATE_DRAWS`.
    length : float
        The paired mean half-width ratio to (A).
    length_high : float
        The 99% upper end of that ratio.
    """

    candidate: str
    configuration: str
    law: str
    clusters: int
    pair: PairedCounts
    hits: int
    shown: int
    length: float = 1.0
    length_high: float = 1.0

    @property
    def calibrated(self) -> bool:
        """The calibration reading of the cell."""
        return calibrated(self.hits, self.shown)


@dataclasses.dataclass(frozen=True)
class Decision:
    """The part (c) decision for one family.

    Parameters
    ----------
    admissible : dict
        Each candidate's admissibility count, or ``None`` when it is not admissible.
    calibrated_counts : dict
        Each admissible candidate's number of calibrated counts at or above its count.
    winner : str
        The candidate the declared order picks.
    """

    admissible: dict[str, int | None]
    calibrated_counts: dict[str, int]
    winner: str


def admissibility(
    cells: Sequence[CandidateCell],
    candidate: str,
    *,
    start: int,
    level: float = LEVEL,
) -> bool:
    """Whether ``candidate`` passes both non-inferiority clauses at and above ``start``.

    Clause (i): each cell's 99% lower end above :data:`PER_CELL_MARGIN`. Clause (ii): the
    pooled 99% lower end over every such cell above :data:`POOLED_MARGIN`. A candidate with
    no cell there is not admissible.
    """
    mine = [cell for cell in cells if cell.candidate == candidate and cell.clusters >= start]
    if not mine:
        return False
    per_cell = all(
        difference_interval([cell.pair], level=level).low > PER_CELL_MARGIN for cell in mine
    )
    pooled = difference_interval([cell.pair for cell in mine], level=level).low > POOLED_MARGIN
    return bool(per_cell and pooled)


def reference_decision(
    cells: Sequence[CandidateCell],
    *,
    floor_a: int,
    jackknife_start: int,
    level: float = LEVEL,
) -> Decision:
    """The declared part (c) decision for one family (plan 5.7, rules 1 and 2).

    (A) is admissible from ``floor_a``. (B) is admissible from ``floor_a`` and (C) from
    ``max(floor_a, jackknife_start)`` when it passes both clauses there. Among admissible
    candidates the one with at least :data:`CALIBRATION_LEAD` more calibrated counts (each
    count calibrated in every configuration and law) wins; otherwise the shortest one whose
    length ratio's upper end lies below 1; ties go to A, then B, then C.
    """
    starts = {
        REFERENCE_ARM: floor_a,
        WANG_ARM: floor_a,
        JACKKNIFE_ARM: max(floor_a, jackknife_start),
    }
    admissible: dict[str, int | None] = {REFERENCE_ARM: floor_a}
    for candidate in (WANG_ARM, JACKKNIFE_ARM):
        ok = admissibility(cells, candidate, start=starts[candidate], level=level)
        admissible[candidate] = starts[candidate] if ok else None
    counts: dict[str, int] = {}
    for candidate, start in admissible.items():
        if start is None:
            continue
        by_count: dict[int, bool] = {}
        for cell in cells:
            if cell.candidate == candidate and cell.clusters >= start:
                by_count[cell.clusters] = by_count.get(cell.clusters, True) and cell.calibrated
        counts[candidate] = sum(by_count.values())
    order = [name for name in CANDIDATE_ARMS if name in counts]
    best = max(counts.values())
    leaders = [name for name in order if counts[name] == best]
    runner_up = max((counts[name] for name in order if name not in leaders), default=-1)
    if len(leaders) == 1 and best - runner_up >= CALIBRATION_LEAD:
        return Decision(admissible, counts, leaders[0])
    contenders = [name for name in order if best - counts[name] < CALIBRATION_LEAD]
    shorter = []
    for name in contenders:
        if name == REFERENCE_ARM:
            continue
        mine = [
            cell
            for cell in cells
            if cell.candidate == name and cell.clusters >= (admissible[name] or 0)
        ]
        if mine and all(cell.length_high < 1.0 for cell in mine):
            shorter.append((float(np.median([cell.length for cell in mine])), name))
    if shorter:
        return Decision(admissible, counts, min(shorter)[1])
    return Decision(admissible, counts, contenders[0])


# ------------------------------------------------------------------------------------ sizing


def g1_threshold(replicates: int, *, floor: float = G1_FLOOR, level: float = LEVEL) -> int:
    """The fewest covered draws of ``replicates`` whose exact lower end reaches ``floor``."""
    low, high = 0, replicates
    while low < high:
        middle = (low + high) // 2
        if clopper_pearson(middle, replicates, confidence_level=level).low >= floor:
            high = middle
        else:
            low = middle + 1
    return low


def g1_pass_probability(coverage: float, replicates: int) -> float:
    """The plug-in probability that G1 passes at an all-draw coverage of ``coverage``."""
    threshold = g1_threshold(replicates)
    return float(stats.binom.sf(threshold - 1, replicates, coverage))


def predictive_g1(hits: int, trials: int, replicates: int) -> float:
    """G1's pass probability with the pilot's Jeffreys posterior of the coverage integrated out."""
    threshold = g1_threshold(replicates)
    a, b = hits + 0.5, trials - hits + 0.5
    k = np.arange(threshold, replicates + 1)
    log_choose = gammaln(replicates + 1) - gammaln(k + 1) - gammaln(replicates - k + 1)
    log_terms = log_choose + betaln(k + a, replicates - k + b) - betaln(a, b)
    return float(np.exp(log_terms).sum())


def bias_pass_probability(standardized_bias: float, replicates: int, margin: float = 0.25) -> float:
    """G2's plug-in pass probability: the bias's 99% interval within ``margin`` SD.

    The interval of a standardized mean bias over ``replicates`` draws has half-width about
    :math:`z_{0.995}/\\sqrt{R}`; both ends must lie inside ``(-margin, margin)``.
    """
    z = float(stats.norm.ppf(0.995))
    se = 1.0 / math.sqrt(replicates)
    upper = stats.norm.cdf((margin - z * se - standardized_bias) / se)
    lower = stats.norm.cdf((-margin + z * se - standardized_bias) / se)
    return float(max(upper - lower, 0.0))


def control_pass_probability(
    ratio: float, ratio_sd: float, positive_high_sd: tuple[float, float], replicates: int
) -> float:
    """Both control clauses' plug-in pass probability at ``replicates`` draws.

    ``ratio`` and ``ratio_sd`` are the pilot's paired IID/cluster ratio and its standard error
    at the pilot's draws, already rescaled to ``replicates``; clause 1 passes when the 99%
    upper end lies below 0.80. ``positive_high_sd`` is the positive arm's SE ratio and its
    standard error at ``replicates``; clause 2 passes when its 99% upper end is at most 1.20.
    The clauses are treated as independent.
    """
    z = float(stats.norm.ppf(0.995))
    clause1 = (
        stats.norm.cdf((0.80 - ratio - z * ratio_sd) / ratio_sd)
        if ratio_sd > 0
        else float(ratio + 0 < 0.80)
    )
    se_ratio, se_sd = positive_high_sd
    clause2 = (
        stats.norm.cdf((1.20 - se_ratio - z * se_sd) / se_sd)
        if se_sd > 0
        else float(se_ratio <= 1.20)
    )
    return float(clause1 * clause2)


def expected_floor(coverage: Mapping[int, Sequence[float]], *, anchor: int) -> int:
    """:math:`\\hat F_f`: the smallest count from which every cell's pilot coverage is at least
    :data:`EXPECTED_FLOOR_COVERAGE` (M5); ``anchor`` when no count below it qualifies."""
    floor = anchor
    for count in sorted(coverage, reverse=True):
        if min(coverage[count]) < EXPECTED_FLOOR_COVERAGE:
            break
        if count <= anchor:
            floor = count
    return floor


def family_pass_probability(
    cells: Sequence[Mapping[str, float]],
    replicates: int,
    *,
    rescale: Callable[[Mapping[str, float], int], Mapping[str, float]] | None = None,
) -> float:
    """The plug-in probability that every cell passes G1, G2 and both control clauses.

    Each cell mapping holds ``coverage`` (all-draw), ``standardized_bias``, ``ratio`` and
    ``ratio_sd`` (the control's paired ratio and its SE at the pilot's ``pilot_draws``),
    ``se_ratio`` and ``se_ratio_sd`` (the positive arm's, likewise) and ``pilot_draws``.
    Standard errors scale as :math:`\\sqrt{n_\\text{pilot}/R}`. Cells are independent.
    """
    product = 1.0
    for cell in cells:
        values = dict(cell) if rescale is None else dict(rescale(cell, replicates))
        scale = math.sqrt(values["pilot_draws"] / replicates)
        product *= g1_pass_probability(values["coverage"], replicates)
        product *= bias_pass_probability(values["standardized_bias"], replicates)
        product *= control_pass_probability(
            values["ratio"],
            values["ratio_sd"] * scale,
            (values["se_ratio"], values["se_ratio_sd"] * scale),
            replicates,
        )
    return float(product)


def required_replicates(cells: Sequence[Mapping[str, float]], *, start: int = 4_000) -> int:
    """The smallest R from ``start`` in steps of :data:`R_STEP`, capped at :data:`R_CAP`,
    whose family product reaches :data:`FAMILY_PASS_TARGET` (M5)."""
    replicates = start
    while replicates < R_CAP and family_pass_probability(cells, replicates) < FAMILY_PASS_TARGET:
        replicates += R_STEP
    return replicates


@dataclasses.dataclass(frozen=True)
class FamilySizing:
    r"""The pilot-based sizing of one floor family (M5).

    Parameters
    ----------
    expected_floor : int
        :math:`\hat F_f` (:func:`expected_floor`).
    plugin : float
        The plug-in family pass probability at :attr:`replicates`.
    predictive : float
        The predictive G1 family probability at :attr:`replicates` (reported, review 2 m4).
    replicates : int
        The R the rule chooses (:func:`required_replicates`).
    predicted_red : tuple of tuple
        ``(configuration, law, count)`` cells at or above the expected floor whose own plug-in
        pass probability at :attr:`replicates` is below 0.90.
    """

    expected_floor: int
    plugin: float
    predictive: float
    replicates: int
    predicted_red: tuple[tuple[str, str, int], ...]


_Z99: Final[float] = float(stats.norm.ppf(0.995))


def sizing_inputs(positive: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, float]:
    """One cell's sizing inputs off its pilot summary rows (positive and control)."""

    def sd(row: Mapping[str, Any], low: str, high: str) -> float:
        return float((float(row[high]) - float(row[low])) / (2.0 * _Z99))

    return {
        "coverage": float(positive["all_draw_coverage"]),
        "hits": float(positive["all_draw_coverage"]) * float(positive["replicates"]),
        "standardized_bias": float(positive["standardized_bias"]),
        "ratio": float(control["paired_se_ratio"]),
        "ratio_sd": sd(control, "paired_se_ratio_ci_lower", "paired_se_ratio_ci_upper"),
        "se_ratio": float(positive["se_ratio"]),
        "se_ratio_sd": sd(positive, "se_ratio_ci_lower", "se_ratio_ci_upper"),
        "pilot_draws": float(positive["replicates"]),
    }


def family_sizing(
    cells: Mapping[tuple[str, str, int], Mapping[str, float]], *, anchor: int
) -> FamilySizing:
    """The M5 sizing of one family from its cells' :func:`sizing_inputs`."""
    by_count: dict[int, list[float]] = {}
    for (_, _, count), values in cells.items():
        by_count.setdefault(count, []).append(values["coverage"])
    start = expected_floor(by_count, anchor=anchor)
    above = {key: values for key, values in cells.items() if key[2] >= start}
    replicates = required_replicates(list(above.values()))
    plugin = family_pass_probability(list(above.values()), replicates)
    predictive = float(
        np.prod(
            [
                predictive_g1(round(v["hits"]), int(v["pilot_draws"]), replicates)
                for v in above.values()
            ]
        )
    )
    red = tuple(
        sorted(
            key
            for key, values in above.items()
            if family_pass_probability([values], replicates) < 0.90
        )
    )
    return FamilySizing(start, plugin, predictive, replicates, red)


def admissibility_probability(
    discordance: float, cells: int, draws: int = CANDIDATE_DRAWS
) -> float:
    """The probability that a candidate exactly as good as (A) passes clause (i) in every cell.

    ``discordance`` is the share of draws covered by exactly one of the two, split evenly.
    Computed by the normal form of :func:`difference_interval`.
    """
    z = float(stats.norm.ppf(0.5 + LEVEL / 2.0))
    se = math.sqrt(discordance / draws)
    one = stats.norm.cdf((0.0 - PER_CELL_MARGIN - z * se) / se) if se > 0 else 1.0
    return float(one**cells)


__all__: Sequence[str] = (
    "CANDIDATE_DRAWS",
    "FAMILY",
    "CandidateCell",
    "Decision",
    "PairedCounts",
    "Reading",
    "admissibility",
    "arm_row",
    "count_passes",
    "difference_interval",
    "family_floor",
    "g1_pass_probability",
    "jackknife_reading",
    "paired_counts",
    "reference_decision",
    "reference_readings",
)
