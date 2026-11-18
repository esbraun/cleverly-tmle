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
#: ``clustered-few-cluster-tmle`` (``property_verdicts.few_cluster_rule``) judges it.
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
#: The **property** bound: the share of a property cell's requested replications the
#: accounting lets a run lose (each properties module's ``FAILURE_BOUND``). The plan asks for
#: 0.10; the harness caps a bound at 0.05
#: (:data:`~tests.studies.evidence.property_failures.MAXIMUM_BOUND`). A lost fit is a miss in
#: G1 either way.
FAILURE_BOUND: Final[float] = 0.05
#: The **primary** bound of both records (``StudyRecord.failure_bound``), the 0.01 every
#: registered study declares (``tests/unit/test_failure_accounting.py::TestPrimaryBound``).
#: The primaries call the fit directly, so a withheld estimate there is a row, not a loss.
PRIMARY_FAILURE_BOUND: Final[float] = 0.01

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
    "jackknife_error",
)
#: The withheld trigger of each withheld package status (:func:`trigger_of`); any other
#: withheld status reads ``"other"``.
STATUS_TRIGGERS: Final[dict[str, str]] = {
    "constant_arm_cell_plugin": "constant_arm_cell",
    "constant_node_plugin": "constant_node",
    "out_of_range_plugin": "out_of_range",
    "jackknife_refit_plugin": "jackknife_refit_plugin",
    "no_cluster_variance_plugin": "no_cluster_variance",
    "stack_failure_plugin": "stack_failure",
}
#: The withheld triggers a row can record; ``""`` is a shown interval. Beyond the package
#: statuses: (B) undefined at :math:`J \le p`, (C) refused before any refit, and (C)'s fit
#: raising on a draw where the main fit ran (its class and message in ``jackknife_error``).
TRIGGERS: Final[tuple[str, ...]] = (
    "",
    *STATUS_TRIGGERS.values(),
    "wang_undefined",
    "jackknife_refused",
    "jackknife_error",
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
    return STATUS_TRIGGERS.get(str(estimate.inference), "other")


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
    error : str
        The class and message of an exception (C)'s fit raised, else ``""``.
    """

    psi: float
    std_error: float
    critical: float
    reference_df: float
    trigger: str = ""
    failed_refits: int = 0
    error: str = ""

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

    A refit failure or any other withheld status is a miss with its trigger. An estimate that
    withholds its interval before any refit (a constant arm, say) carries no jackknife record;
    (C) then reads withheld with the main fit's own trigger, as (A) does (review B1).
    """
    record = estimate.jackknife
    if record is None:
        trigger = trigger_of(estimate)
        if not trigger:
            raise AssertionError("a shown (C) estimate carries no jackknife record")
        return Reading(float(estimate.psi), float("nan"), float("nan"), clusters - 2, trigger)
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


def failed_jackknife(psi: float, clusters: int, error: BaseException) -> Reading:
    """(C) on a draw where its fit raised and the main fit did not: a miss, the error kept."""
    message = f"{type(error).__name__}: {error}"[:160]
    return Reading(psi, float("nan"), float("nan"), clusters - 2, "jackknife_error", error=message)


def error_text(error: BaseException) -> str:
    """The class and message of an exception, as a row records it."""
    return f"{type(error).__name__}: {error}"[:160]


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
        "jackknife_error": reading.error,
    }


#: The package-GLM columns an ``ltmle_in_sample`` positive row carries.
PACKAGE_COLUMNS: Final[tuple[str, ...]] = (
    "package_glm_raised",
    "package_glm_shown",
    "package_glm_covered",
    "package_glm_error",
)
#: The pilot's timing columns (:func:`timed`): a key's CPU seconds, and the package-GLM
#: refit's share of them on an LTMLE positive row.
TIMING_COLUMNS: Final[tuple[str, ...]] = ("cpu_seconds", "package_glm_cpu_seconds")


def timed(
    fit_key: Callable[[Any, Any], list[dict[str, Any]]],
) -> Callable[[Any, Any], list[dict[str, Any]]]:
    """``fit_key`` with its process CPU seconds written on every row it returns (the pilot's)."""
    import time

    def run(key: Any, drawn: Any) -> list[dict[str, Any]]:
        started = time.process_time()
        rows = fit_key(key, drawn)
        spent = time.process_time() - started
        return [{**row, "cpu_seconds": spent} for row in rows]

    return run


def recording_every_error(
    fit_key: Callable[[Any, Any], list[dict[str, Any]]],
) -> Callable[[Any, Any], list[dict[str, Any]]]:
    """``fit_key`` with every exception recorded as a lost key (the pilot's, never a run's).

    The pilot is diagnostic: a key that raises anything but ``MemoryError`` or an interrupt is
    carried to the harness as already recorded, with its class, call and message, so
    ``pilot-failures.csv`` lists it and the pilot never stops on one key. A declared run keeps
    :func:`cleverly.validation._failures.recordable`, where an unknown class stops the run.
    """
    import types

    from cleverly.validation._failures import failure_call
    from tests.studies.evidence.replication import _Recorded

    def run(key: Any, drawn: Any) -> list[dict[str, Any]]:
        try:
            return fit_key(key, drawn)
        except MemoryError:
            raise
        except Exception as error:
            failure = types.SimpleNamespace(
                call=failure_call(error), error_type=type(error).__name__, message=str(error)
            )
            raise _Recorded(failure) from error

    return run


#: The readings :func:`floor_readings` adds to every row of the family.
READING_COLUMNS: Final[tuple[str, ...]] = (
    "clusters",
    "requested_draws",
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
    "failed_refit_share",
)


def parse(cell: str) -> tuple[str, str, int, str]:
    """``(fit, law, J, arm)`` of a published cell name."""
    fit, law, count, arm = cell.split("__")
    return fit, law, int(count.removeprefix("j")), arm


def floor_readings(summary: pd.DataFrame, rows: pd.DataFrame, record: Any) -> None:
    """Fill :data:`READING_COLUMNS` on every row of ``few_cluster_reference`` and of
    ``few_cluster_candidate`` (whose (C) rows carry the package-GLM split, m6)."""
    level = record.margins.confidence_level
    summary["withheld_by_trigger"] = summary["withheld_by_trigger"].astype(object)
    for family in (FAMILY, CANDIDATE_FAMILY):
        _family_readings(summary, rows, family, record, level)


def _family_readings(
    summary: pd.DataFrame, rows: pd.DataFrame, family_name: str, record: Any, level: float
) -> None:
    family = rows.loc[rows["property"] == family_name]
    groups = dict(tuple(family.groupby("cell", sort=False)))
    for index in summary.index[summary["property"] == family_name]:
        name = str(summary.loc[index, "cell"])
        _, _, clusters, arm = parse(name)
        group = groups[name]
        shown = group.loc[group["withheld"] == 0]
        hits, trials = int(shown["covered"].sum()), len(shown)
        wanted = int(group["requested_replicates"].iloc[0])
        triggers = group.loc[group["withheld"] == 1, "trigger"].value_counts().sort_index()
        values: dict[str, Any] = {
            "clusters": float(clusters),
            "requested_draws": float(wanted),
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
        if "package_glm_raised" in group:
            values.update(package_readings(group, level))
        if arm == JACKKNIFE_ARM:
            refits = pd.to_numeric(group["failed_refits"], errors="coerce").fillna(0)
            values["failed_refit_share"] = float((refits > 0).sum() / wanted)
            reference = groups.get(name.removesuffix(JACKKNIFE_ARM) + reference_for(JACKKNIFE_ARM))
            if reference is not None:
                values.update(jackknife_package_split(group, reference))
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


def jackknife_package_split(jackknife: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """(C)'s shown-only coverage split by whether the package GLM raised on the draw (m6).

    The package-GLM flag lives on the (A) row of the same draw. A separated last iterate of
    the study GLM is outside the jackknife's smooth-equation argument, so (C) on those draws
    is published apart.
    """
    if "package_glm_raised" not in reference:
        return {}
    flags = pd.to_numeric(reference.set_index("replicate")["package_glm_raised"], errors="coerce")
    if flags.isna().all():
        return {}
    raised = jackknife["replicate"].map(flags).fillna(0).astype(int)
    shown = jackknife["withheld"] == 0
    out = {}
    for label, mask in (("raised", raised == 1), ("ran", raised == 0)):
        part = jackknife.loc[mask & shown]
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
    gain_low : float
        The 99% lower end of the cell's paired coverage gain over (A), the
        ``few_cluster_candidate`` verdict row's S7 interval (clause (i)); ``nan`` when the
        candidate has no verdict row there.
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
    gain_low: float = float("nan")

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
    reasons : dict
        Why each inadmissible candidate is not admissible, the measured reason a refusal names.
    """

    admissible: dict[str, int | None]
    calibrated_counts: dict[str, int]
    winner: str
    reasons: dict[str, str] = dataclasses.field(default_factory=dict)


def admissibility(
    cells: Sequence[CandidateCell],
    candidate: str,
    *,
    start: int,
    pooled: Mapping[tuple[str, str], bool],
) -> tuple[bool, str]:
    """Whether ``candidate`` passes both non-inferiority clauses at and above ``start``.

    Clause (i): each cell's ``few_cluster_candidate`` verdict, the 99% lower end of its paired
    coverage gain at least :data:`PER_CELL_MARGIN` (the S7 interval, :func:`candidate_verdicts`).
    Clause (ii): every configuration's ``few_cluster_candidate_pooled`` verdict at this start
    (:func:`pooled_rows`), each above :data:`POOLED_MARGIN`. Running clause (ii) per
    configuration and taking the conjunction is plan 5.7's "the worse of the two" (rev 3.6, m9).
    A candidate with no cell there is not admissible.

    Returns
    -------
    tuple
        The verdict and, when it fails, the measured reason.
    """
    mine = [cell for cell in cells if cell.candidate == candidate and cell.clusters >= start]
    if not mine:
        return False, f"no {candidate} cell at or above {start} clusters"
    failing = [
        f"{cell.configuration}/{cell.law}/J{cell.clusters}"
        for cell in mine
        if not cell.gain_low >= PER_CELL_MARGIN
    ]
    configurations = sorted({cell.configuration for cell in mine})
    unpooled = [name for name in configurations if not pooled.get((name, candidate), False)]
    reasons = []
    if failing:
        reasons.append(f"clause (i) fails at {', '.join(failing)}")
    if unpooled:
        reasons.append(f"clause (ii) fails in {', '.join(unpooled)}")
    return not reasons, "; ".join(reasons)


def reference_decision(
    cells: Sequence[CandidateCell],
    *,
    floor_a: int,
    jackknife_start: int,
    pooled: Mapping[tuple[str, str], bool],
) -> Decision:
    """The declared part (c) decision for one family (plan 5.7, rules 1 and 2).

    (A) is admissible from ``floor_a``. (B) is admissible from ``floor_a`` and (C) from
    ``max(floor_a, jackknife_start)`` when it passes both clauses there (:func:`admissibility`).
    Among admissible candidates the one with at least :data:`CALIBRATION_LEAD` more calibrated
    counts (each count calibrated in every configuration and law) wins; otherwise the shortest
    one whose length ratio's upper end lies below 1; ties go to A, then B, then C.
    """
    starts = {
        REFERENCE_ARM: floor_a,
        WANG_ARM: floor_a,
        JACKKNIFE_ARM: max(floor_a, jackknife_start),
    }
    admissible: dict[str, int | None] = {REFERENCE_ARM: floor_a}
    reasons: dict[str, str] = {}
    for candidate in (WANG_ARM, JACKKNIFE_ARM):
        ok, reason = admissibility(cells, candidate, start=starts[candidate], pooled=pooled)
        admissible[candidate] = starts[candidate] if ok else None
        if not ok:
            reasons[candidate] = reason
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
        return Decision(admissible, counts, leaders[0], reasons)
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
        return Decision(admissible, counts, min(shorter)[1], reasons)
    return Decision(admissible, counts, contenders[0], reasons)


# ------------------------------------------------------------------ the candidate verdict family

#: The verdict family of part (c) (review M1): each candidate cell (role ``positive``) paired
#: with a copy of (A)'s row on the same draw (role ``control``), on the first
#: :data:`CANDIDATE_DRAWS` draws.
CANDIDATE_FAMILY: Final[str] = "few_cluster_candidate"
#: One summary row per configuration and candidate: clause (ii) over the cells at or above the
#: candidate's admissibility count.
POOLED_FAMILY: Final[str] = "few_cluster_candidate_pooled"
#: The summary columns the candidate and pooled rows publish.
CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
    "coverage_gain_ci_lower",
    "coverage_gain_ci_upper",
    "pooled_start",
    "pooled_cells",
)
#: The candidates with a verdict row: (B) and (C).
VERDICT_CANDIDATES: Final[tuple[str, ...]] = (WANG_ARM, JACKKNIFE_ARM)
_REFERENCE_PREFIX: Final[str] = "reference_for_"


def reference_for(candidate: str) -> str:
    """The arm name of (A)'s copy that a candidate cell is paired with."""
    return f"{_REFERENCE_PREFIX}{candidate}"


def positive_of(cell: str) -> str:
    """The candidate cell of a ``few_cluster_candidate`` pair, given either member."""
    fit, law, count, arm = cell.split("__")
    return "__".join((fit, law, count, arm.removeprefix(_REFERENCE_PREFIX)))


def candidate_copy(
    row: Mapping[str, Any], *, arm: str, role: str, requested: int
) -> dict[str, Any]:
    """A ``few_cluster_reference`` row filed as one arm of a ``few_cluster_candidate`` pair."""
    fit, law, count, _ = str(row["cell"]).split("__")
    return {
        **row,
        "property": CANDIDATE_FAMILY,
        "cell": "__".join((fit, law, count, arm)),
        "role": role,
        "requested_replicates": requested,
    }


def candidate_seed(record: Any, cell: str) -> int:
    """The resampling seed of a pair's S7 interval, the same for both members."""
    from tests.studies.evidence.seeds import stream_seed

    return int(stream_seed(record, CANDIDATE_FAMILY, positive_of(cell)))


def candidate_verdicts(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    record: Any,
    groups: Sequence[Sequence[str]],
) -> None:
    """Clause (i) on every ``few_cluster_candidate`` pair present in ``summary`` (M1).

    The pair's paired coverage gain of the candidate over (A)'s copy
    (:func:`~tests.studies.evidence.properties.coverage_gain_interval`, a percentile bootstrap
    of the mean paired difference on draws both kept) is written on both rows. Each row's own
    verdict holds, and the joint verdict is the gain's lower end at least
    :data:`PER_CELL_MARGIN`. The S7 clause of the module's rule reads the same columns, and
    under a lost draw it fills the candidate uncovered and (A) covered, the worst case for the
    candidate. This replaces the normal interval of :func:`difference_interval` for clause (i)
    (rev 3.6).
    """
    from tests.studies.evidence.properties import coverage_gain_interval

    margins = record.margins
    family = rows.loc[rows["property"] == CANDIDATE_FAMILY]
    by_cell = dict(tuple(family.groupby("cell", sort=False)))
    present = set(summary.loc[summary["property"] == CANDIDATE_FAMILY, "cell"].astype(str))
    for members in groups:
        if not set(members) <= present:
            continue
        positive = positive_of(members[0])
        control = next(name for name in members if name != positive)
        low, high = coverage_gain_interval(
            by_cell[positive],
            by_cell[control],
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=candidate_seed(record, positive),
        )
        mask = (summary["property"] == CANDIDATE_FAMILY) & summary["cell"].isin(members)
        summary.loc[mask, "coverage_gain_ci_lower"] = low
        summary.loc[mask, "coverage_gain_ci_upper"] = high
        summary.loc[mask, "passed"] = True
        summary.loc[mask, "property_passed"] = bool(low >= PER_CELL_MARGIN)


def pooled_name(configuration: str, candidate: str) -> str:
    """The cell of a pooled row."""
    return f"{configuration}__{candidate}"


def pooled_sources(row: Mapping[str, Any], rows: pd.DataFrame) -> list[tuple[str, str]]:
    """The candidate cells a pooled row reads: its configuration's pairs at or above its start."""
    configuration, candidate = str(row["cell"]).split("__")
    start = float(row["pooled_start"])
    present = rows.loc[rows["property"] == CANDIDATE_FAMILY, "cell"].astype(str).unique()
    out = []
    for name in sorted(present):
        fit, _, count, arm = name.split("__")
        if fit != configuration or int(count.removeprefix("j")) < start:
            continue
        if arm in (candidate, reference_for(candidate)):
            out.append((CANDIDATE_FAMILY, name))
    return out


def _pooled_differences(
    rows: pd.DataFrame, cells: Sequence[tuple[str, str]], *, aligned: bool
) -> np.ndarray:
    """The paired differences of every pair in ``cells``, concatenated.

    Complete case: the draws both arms kept. Aligned: every requested draw, a lost candidate
    draw uncovered and a lost (A) draw covered, as the S7 fill.
    """
    family = rows.loc[rows["property"] == CANDIDATE_FAMILY]
    names = {name for _, name in cells}
    out: list[float] = []
    for name in sorted(names):
        if positive_of(name) != name:
            continue
        fit, law, count, arm = name.split("__")
        positive = family.loc[family["cell"] == name]
        control = family.loc[family["cell"] == "__".join((fit, law, count, reference_for(arm)))]
        left = dict(
            zip(positive["replicate"].astype(int), positive["covered"].astype(float), strict=True)
        )
        right = dict(
            zip(control["replicate"].astype(int), control["covered"].astype(float), strict=True)
        )
        if aligned:
            requested = int(
                (positive if len(positive) else control)["requested_replicates"].iloc[0]
            )
            ids: Iterable[int] = range(requested)
            out.extend(left.get(i, 0.0) - right.get(i, 1.0) for i in ids)
        else:
            out.extend(left[i] - right[i] for i in sorted(set(left) & set(right)))
    return np.asarray(out, dtype=float)


def pooled_interval(
    rows: pd.DataFrame, cells: Sequence[tuple[str, str]], *, record: Any, cell: str, aligned: bool
) -> Interval:
    """Clause (ii)'s interval: the percentile bootstrap of the mean pooled paired difference.

    Resampling the concatenated draws ignores which cell a draw came from, which adds the
    spread between cell means to the variance: conservative against the stratified form.
    """
    from tests.studies.evidence.properties import mean_difference_interval
    from tests.studies.evidence.seeds import stream_seed

    differences = _pooled_differences(rows, cells, aligned=aligned)
    if not len(differences):
        return Interval(float("nan"), float("nan"))
    return mean_difference_interval(
        differences,
        replicates=record.margins.bootstrap_replicates,
        confidence_level=record.margins.confidence_level,
        seed=int(stream_seed(record, POOLED_FAMILY, cell)),
    )


def pooled_rows(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    record: Any,
    starts: Mapping[tuple[str, str], int],
) -> list[dict[str, Any]]:
    """One ``few_cluster_candidate_pooled`` row per configuration and candidate in ``starts``.

    ``starts`` maps ``(configuration, candidate)`` to the candidate's admissibility count in
    that configuration's family, read off the same summary's floor verdicts. A row is written
    only when its configuration has a candidate cell. It passes when the complete-case pooled
    gain's 99% lower end lies above :data:`POOLED_MARGIN`.
    """
    present = set(rows.loc[rows["property"] == CANDIDATE_FAMILY, "cell"].astype(str))
    configurations = {name.split("__")[0] for name in present}
    out = []
    for (configuration, candidate), start in sorted(starts.items()):
        if configuration not in configurations:
            continue
        name = pooled_name(configuration, candidate)
        row: dict[str, Any] = dict.fromkeys(summary.columns, np.nan)
        row.update(property=POOLED_FAMILY, cell=name, role="positive", pooled_start=float(start))
        cells = pooled_sources(row, rows)
        interval = pooled_interval(rows, cells, record=record, cell=name, aligned=False)
        passed = bool(interval.low > POOLED_MARGIN)
        row.update(
            coverage_gain_ci_lower=interval.low,
            coverage_gain_ci_upper=interval.high,
            pooled_cells=float(len(cells) // 2),
            passed=passed,
            property_passed=passed,
        )
        out.append(row)
    return out


def candidate_rule() -> Any:
    """The declared rule of the two part (c) families (review M1).

    ``few_cluster_candidate``: the harness's S7 clause, the gain's 99% lower end at least
    :data:`PER_CELL_MARGIN`, on each pair (group ``pair``). ``few_cluster_candidate_pooled``:
    the pooled gain's lower end above :data:`POOLED_MARGIN`, read off the row and, under a
    lost draw, recomputed with the S7 fill over :func:`pooled_sources`.
    """
    from tests.studies.evidence import property_failures as pf

    def judge(interval: Interval) -> Any:
        return pf.Judged(interval.low - POOLED_MARGIN, bool(interval.low > POOLED_MARGIN))

    def read(ctx: Any, row: pd.Series) -> Any:
        return judge(
            Interval(float(row["coverage_gain_ci_lower"]), float(row["coverage_gain_ci_upper"]))
        )

    def reads(ctx: Any, row: pd.Series) -> list[tuple[str, str]]:
        return pooled_sources(row, ctx.rows)

    def candidates(ctx: Any, row: pd.Series) -> list[tuple[str, Any]]:
        interval = pooled_interval(
            ctx.rows, reads(ctx, row), record=ctx.record, cell=str(row["cell"]), aligned=True
        )
        return [("lost candidate draws uncovered, lost reference draws covered", judge(interval))]

    def is_pooled(ctx: Any, row: pd.Series) -> bool:
        return str(row["property"]) == POOLED_FAMILY

    def is_pair(ctx: Any, row: pd.Series) -> bool:
        return str(row["property"]) == CANDIDATE_FAMILY

    return pf.FamilyRule(
        families=frozenset({CANDIDATE_FAMILY, POOLED_FAMILY}),
        shared="candidate_verdicts",
        clauses=(
            pf.coverage_gain_clause(
                "candidate coverage gain over the reference",
                floor=PER_CELL_MARGIN,
                seed=lambda ctx, row: candidate_seed(ctx.record, str(row["cell"])),
                applies=is_pair,
                group="pair",
            ),
            pf.Clause(
                "pooled candidate coverage gain over the reference",
                "S7p",
                is_pooled,
                read,
                candidates,
                target="passed",
                imputation=pf.SUPPORT,
                reads=reads,
            ),
        ),
        group="pair",
        sources=lambda row, rows: (
            pooled_sources(row, rows)
            if str(row["property"]) == POOLED_FAMILY
            else [(str(row["property"]), str(row["cell"]))]
        ),
    )


# ------------------------------------------------------------------- red owners and the outcome

#: The roadmap owner of every declared red reading of both studies (``docs/roadmap.md``,
#: "Red-cell owners"); ``tests/unit/test_few_cluster_floor_design.py`` asserts it is listed.
RED_OWNER: Final[str] = "X54-cluster-floor"
#: A red positive cell below its family's floor whose withheld and lost draws explain it: the
#: withheld-and-lost share above :data:`AVAILABILITY_SHARE`, the shown-only exact interval
#: holding 0.95 and the SE-ratio interval holding 1.
AVAILABILITY_READING: Final[str] = "below floor (availability)"
#: Any other red positive cell below its family's floor.
FINITE_SAMPLE_READING: Final[str] = "below floor (finite-sample)"
#: A red IID control below its family's floor: four to nine clusters cannot show the
#: IID-to-cluster SE ratio below 0.80 (plan 5.1, clause 1), or the control turns red jointly
#: with its positive because the positive's shown-only SE-ratio upper end exceeds 1.20
#: (clause 2), or the cell has fewer than two shown draws.
CONTROL_READING: Final[str] = "control power"
#: A red at or above the anchor whose recorded diagnosis (:data:`NO_DEFECT_DIAGNOSES`) finds no
#: defect and which agrees with its committed counterpart (:func:`agrees_with_committed`),
#: cited against the pilot's predictive family figure.
CHANCE_READING: Final[str] = "finite-sample (plug-in sizing)"
#: A red ``few_cluster_candidate`` or pooled row: the candidate is not non-inferior to (A)
#: there, which the part (c) decision reads (rules 4 and 5).
CANDIDATE_READING: Final[str] = "candidate not non-inferior"
READINGS: Final[tuple[str, ...]] = (
    AVAILABILITY_READING,
    FINITE_SAMPLE_READING,
    CONTROL_READING,
    CHANCE_READING,
    CANDIDATE_READING,
)
#: The withheld-and-lost share above which a red reads as availability.
AVAILABILITY_SHARE: Final[float] = 0.03
#: Recorded no-defect diagnoses of reds at or above an anchor, by ``family/cell``; empty at the
#: declaration, so such a red is a defect suspect until a diagnosis is recorded here.
NO_DEFECT_DIAGNOSES: Final[dict[str, str]] = {}


@dataclasses.dataclass(frozen=True)
class Counterpart:
    """A committed cell an anchor red is compared with (review B3).

    Parameters
    ----------
    slug : str
        The committed study.
    hits : int
        A positive cell's covering draws (every draw; the committed cells lost none).
    draws : int
        A positive cell's draws.
    ratio : float
        A control cell's paired IID/cluster SE ratio.
    ratio_low : float
        Its 99% interval's lower end.
    ratio_high : float
        Its 99% interval's upper end.
    """

    slug: str
    hits: int = 0
    draws: int = 0
    ratio: float = math.nan
    ratio_low: float = math.nan
    ratio_high: float = math.nan


#: The committed ``few_cluster_reference`` cells of ``clustered-few-cluster-tmle`` and
#: ``few-cluster-cross-fitted-ltmle`` at 10, 20 and 30 clusters, copied from their committed
#: ``properties.csv``; ``tests/unit/test_few_cluster_floor_design.py`` checks the copy.
COMMITTED_CELLS: Final[dict[str, Counterpart]] = {
    "drtmle_crossfit__equal10__j10__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7659136930500346,
        ratio_low=0.7585610279452084,
        ratio_high=0.7735280361774681,
    ),
    "drtmle_crossfit__equal10__j10__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3791, draws=4000
    ),
    "drtmle_crossfit__equal10__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7338898342816227,
        ratio_low=0.7298065056276075,
        ratio_high=0.7379010837958329,
    ),
    "drtmle_crossfit__equal10__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3810, draws=4000
    ),
    "drtmle_crossfit__equal10__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7277934116762391,
        ratio_low=0.7248025319161029,
        ratio_high=0.7307729986303725,
    ),
    "drtmle_crossfit__equal10__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3814, draws=4000
    ),
    "drtmle_crossfit__unequal_informative__j10__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.6837715997986044,
        ratio_low=0.6749718925869178,
        ratio_high=0.6928792757092729,
    ),
    "drtmle_crossfit__unequal_informative__j10__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3732, draws=4000
    ),
    "drtmle_crossfit__unequal_informative__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.6229619023260806,
        ratio_low=0.6188893618001229,
        ratio_high=0.6269583930655988,
    ),
    "drtmle_crossfit__unequal_informative__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3788, draws=4000
    ),
    "drtmle_crossfit__unequal_informative__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.6088131653225648,
        ratio_low=0.6058321923239623,
        ratio_high=0.6117560276688337,
    ),
    "drtmle_crossfit__unequal_informative__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3784, draws=4000
    ),
    "ltmle_in_sample__equal10__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7140671065474723,
        ratio_low=0.7102849331131998,
        ratio_high=0.717806595480112,
    ),
    "ltmle_in_sample__equal10__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3764, draws=4000
    ),
    "ltmle_in_sample__equal10__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7149115724437758,
        ratio_low=0.7118863585392204,
        ratio_high=0.7179822264359734,
    ),
    "ltmle_in_sample__equal10__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3797, draws=4000
    ),
    "ltmle_in_sample__unequal_informative__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.5894088568271826,
        ratio_low=0.5862444657530322,
        ratio_high=0.5925897684794095,
    ),
    "ltmle_in_sample__unequal_informative__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3772, draws=4000
    ),
    "ltmle_in_sample__unequal_informative__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.5886622544223854,
        ratio_low=0.5862071894827089,
        ratio_high=0.5911276920115287,
    ),
    "ltmle_in_sample__unequal_informative__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3779, draws=4000
    ),
    "tmle_crossfit__equal10__j10__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7340648295289189,
        ratio_low=0.728161204722442,
        ratio_high=0.7401980927257343,
    ),
    "tmle_crossfit__equal10__j10__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3740, draws=4000
    ),
    "tmle_crossfit__equal10__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7226555263996368,
        ratio_low=0.7188286710436261,
        ratio_high=0.7265402481174634,
    ),
    "tmle_crossfit__equal10__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3770, draws=4000
    ),
    "tmle_crossfit__equal10__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7204743247026979,
        ratio_low=0.717436205334585,
        ratio_high=0.72348308684362,
    ),
    "tmle_crossfit__equal10__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3807, draws=4000
    ),
    "tmle_crossfit__unequal_informative__j10__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.6112508725800283,
        ratio_low=0.6061957183926808,
        ratio_high=0.6163240771484058,
    ),
    "tmle_crossfit__unequal_informative__j10__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3724, draws=4000
    ),
    "tmle_crossfit__unequal_informative__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.5932922794766998,
        ratio_low=0.59019248779312,
        ratio_high=0.5965418249533712,
    ),
    "tmle_crossfit__unequal_informative__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3781, draws=4000
    ),
    "tmle_crossfit__unequal_informative__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.5909727783147798,
        ratio_low=0.5885117587603365,
        ratio_high=0.5934964190047076,
    ),
    "tmle_crossfit__unequal_informative__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3788, draws=4000
    ),
    "tmle_cv_evaluation__equal10__j10__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7583734079833067,
        ratio_low=0.7492299473764898,
        ratio_high=0.7674192252617587,
    ),
    "tmle_cv_evaluation__equal10__j10__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3762, draws=4000
    ),
    "tmle_cv_evaluation__equal10__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7339316805564021,
        ratio_low=0.7292848016182738,
        ratio_high=0.7386259064011245,
    ),
    "tmle_cv_evaluation__equal10__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3780, draws=4000
    ),
    "tmle_cv_evaluation__equal10__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7273102443207347,
        ratio_low=0.7238037634039721,
        ratio_high=0.7307511904428045,
    ),
    "tmle_cv_evaluation__equal10__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3804, draws=4000
    ),
    "tmle_in_sample__equal10__j10__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7169291865059745,
        ratio_low=0.7110246049114152,
        ratio_high=0.7226718058978878,
    ),
    "tmle_in_sample__equal10__j10__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3721, draws=4000
    ),
    "tmle_in_sample__equal10__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.7143718977123157,
        ratio_low=0.7106475665053341,
        ratio_high=0.7181875018684405,
    ),
    "tmle_in_sample__equal10__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3760, draws=4000
    ),
    "tmle_in_sample__equal10__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.715267251083649,
        ratio_low=0.7122405786644733,
        ratio_high=0.7182689815536448,
    ),
    "tmle_in_sample__equal10__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3800, draws=4000
    ),
    "tmle_in_sample__unequal_informative__j10__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.6019345817235324,
        ratio_low=0.596836410834067,
        ratio_high=0.6068673380455044,
    ),
    "tmle_in_sample__unequal_informative__j10__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3696, draws=4000
    ),
    "tmle_in_sample__unequal_informative__j20__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.5893439345791245,
        ratio_low=0.5861630933142585,
        ratio_high=0.5925473180513177,
    ),
    "tmle_in_sample__unequal_informative__j20__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3773, draws=4000
    ),
    "tmle_in_sample__unequal_informative__j30__iid_t_control": Counterpart(
        "clustered-few-cluster-tmle",
        ratio=0.5885852260666844,
        ratio_low=0.5861149297807821,
        ratio_high=0.5909999574876917,
    ),
    "tmle_in_sample__unequal_informative__j30__t_reference": Counterpart(
        "clustered-few-cluster-tmle", hits=3779, draws=4000
    ),
    "ltmle_crossfit__equal40__j20__iid_t_control": Counterpart(
        "few-cluster-cross-fitted-ltmle",
        ratio=0.6546786399021827,
        ratio_low=0.6509957529909383,
        ratio_high=0.6584282441966791,
    ),
    "ltmle_crossfit__equal40__j20__t_reference": Counterpart(
        "few-cluster-cross-fitted-ltmle", hits=3788, draws=4000
    ),
    "ltmle_crossfit__equal40__j30__iid_t_control": Counterpart(
        "few-cluster-cross-fitted-ltmle",
        ratio=0.6509862800563911,
        ratio_low=0.6479423316124039,
        ratio_high=0.6540562900134884,
    ),
    "ltmle_crossfit__equal40__j30__t_reference": Counterpart(
        "few-cluster-cross-fitted-ltmle", hits=3799, draws=4000
    ),
    "ltmle_crossfit__unequal40__j20__iid_t_control": Counterpart(
        "few-cluster-cross-fitted-ltmle",
        ratio=0.6328463221146452,
        ratio_low=0.6285451106301871,
        ratio_high=0.6371704430410732,
    ),
    "ltmle_crossfit__unequal40__j20__t_reference": Counterpart(
        "few-cluster-cross-fitted-ltmle", hits=3756, draws=4000
    ),
    "ltmle_crossfit__unequal40__j30__iid_t_control": Counterpart(
        "few-cluster-cross-fitted-ltmle",
        ratio=0.6226054143210805,
        ratio_low=0.6192238198241798,
        ratio_high=0.6260680055177177,
    ),
    "ltmle_crossfit__unequal40__j30__t_reference": Counterpart(
        "few-cluster-cross-fitted-ltmle", hits=3770, draws=4000
    ),
}
#: Each floor-study fit and the committed fit of the same family it is compared with at and
#: above the anchor (review B3). An estimated configuration reads the committed oracle cell. The
#: d5 configurations have no committed counterpart.
COUNTERPART_FITS: Final[dict[str, str]] = {
    "tmle_in_sample_known": "tmle_in_sample",
    "tmle_in_sample_estimated": "tmle_in_sample",
    "tmle_stacked_known": "tmle_crossfit",
    "tmle_stacked_estimated": "tmle_crossfit",
    "tmle_fold_evaluated_known": "tmle_cv_evaluation",
    "tmle_fold_evaluated_estimated": "tmle_cv_evaluation",
    "drtmle_known": "drtmle_crossfit",
    "ltmle_in_sample": "ltmle_in_sample",
    "ltmle_cross_fitted": "ltmle_crossfit",
}


def anchor_counterparts() -> dict[str, Counterpart]:
    """Each floor-study cell with a committed counterpart at the same law and count."""
    out = {}
    for name, counterpart in COMMITTED_CELLS.items():
        committed_fit, rest = name.split("__", 1)
        for fit, target in COUNTERPART_FITS.items():
            if target == committed_fit:
                out[f"{fit}__{rest}"] = counterpart
    return out


def agrees_with_committed(
    row: Mapping[str, Any], counterpart: Counterpart | None, *, control: bool
) -> bool:
    """Whether a red cell agrees with its committed counterpart (review B3).

    The two-sample 99% interval of the difference must hold 0: the all-draw coverage of a
    positive (withheld and lost draws as misses) against the committed coverage, by the normal
    form of a difference of two independent proportions; the paired IID/cluster ratio of a
    control against the committed ratio, each SE read off its own 99% interval. No counterpart
    is no agreement.
    """
    if counterpart is None:
        return False
    z = float(stats.norm.ppf(0.5 + LEVEL / 2.0))
    if control:
        mine = (float(row["paired_se_ratio_ci_upper"]) - float(row["paired_se_ratio_ci_lower"])) / (
            2.0 * z
        )
        theirs = (counterpart.ratio_high - counterpart.ratio_low) / (2.0 * z)
        difference = float(row["paired_se_ratio"]) - counterpart.ratio
        spread = math.hypot(mine, theirs)
    else:
        requested = float(row["requested_draws"])
        p1 = float(row["all_draw_coverage"])
        p2 = counterpart.hits / counterpart.draws
        difference = p1 - p2
        spread = math.sqrt(p1 * (1.0 - p1) / requested + p2 * (1.0 - p2) / counterpart.draws)
    return bool(math.isfinite(difference) and abs(difference) <= z * spread)


#: The study's family name and the package's floor family (``FLOOR_FAMILY_NAMES`` keys of
#: :data:`cleverly._inference_status.CLUSTER_INTERVAL_FLOORS`), where they differ (m7).
PACKAGE_FAMILIES: Final[dict[str, str]] = {
    "tmle_in_sample": "tmle_in_sample",
    "tmle_stacked": "tmle_stacked",
    "tmle_fold_evaluated": "tmle_fold_evaluated",
    "drtmle_known": "drtmle_cross_fitted_known_mechanism",
    "ltmle_in_sample": "ltmle_in_sample",
    "ltmle_cross_fitted": "ltmle_cross_fitted",
}
#: The (C) status the package ships for each compared floor family (``Outcome.jackknife``
#: before any reason): the jackknife is an option on the four families part (c) compares, and
#: the post-run test fails until a committed run that changes a status changes this map too.
SHIPPED_JACKKNIFE: Final[dict[str, str]] = {
    "tmle_in_sample": "option",
    "tmle_stacked": "option",
    "ltmle_in_sample": "option",
    "ltmle_cross_fitted": "option",
}
#: The route of a family whose floor moved down from its anchor.
LOWERED: Final[str] = "lowered"
#: The route of a family with a red at or above its anchor that has no committed counterpart
#: (J = 15 or 39, a d5 cell, an LTMLE count of 39): nothing can compare it, so it is never read as
#: chance, the floor stays at the anchor, and the family goes to a redeclaration.
REDECLARATION: Final[str] = "redeclaration"


@dataclasses.dataclass(frozen=True)
class Route:
    """The declared reading of one floor family's verdicts (plan 5.8, review M3).

    Parameters
    ----------
    family : str
        The study's family name.
    floor : int
        :math:`F_f` under :func:`family_floor`.
    route : str
        ``"lowered"``, ``"kept: {count} fails"`` (the count below the anchor fails),
        ``"defect suspect: {counts}"`` (counts at or above the anchor fail; diagnosed first), or
        ``"redeclaration: {counts}"`` (a red at or above the anchor has no committed
        counterpart, :data:`REDECLARATION`).
    owners : dict
        Each red ``family/cell`` and ``(RED_OWNER, reading)``, or ``None`` when no reading
        holds it (a defect suspect).
    """

    family: str
    floor: int
    route: str
    owners: dict[str, tuple[str, str] | None]

    @property
    def defect_suspects(self) -> tuple[str, ...]:
        """The red cells no reading holds."""
        return tuple(sorted(name for name, owner in self.owners.items() if owner is None))


def _contains(row: Mapping[str, Any], low: str, high: str, value: float) -> bool:
    lo, hi = float(row.get(low, math.nan)), float(row.get(high, math.nan))
    return bool(lo <= value <= hi)


def red_reading(positive: Mapping[str, Any]) -> str:
    """The below-floor reading of a red positive row (availability or finite-sample)."""
    requested = float(positive["requested_draws"])
    kept = float(positive["replicates"])
    withheld = float(positive["withheld_draws"])
    share = (withheld + requested - kept) / requested if requested else 0.0
    if (
        share > AVAILABILITY_SHARE
        and _contains(positive, "shown_ci_lower", "shown_ci_upper", 0.95)
        and _contains(positive, "se_ratio_ci_lower", "se_ratio_ci_upper", 1.0)
    ):
        return AVAILABILITY_READING
    return FINITE_SAMPLE_READING


def declared_route(
    family: str,
    cells: Mapping[tuple[str, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    anchor: int,
    diagnoses: Mapping[str, str] | None = None,
    counterparts: Mapping[str, Counterpart] | None = None,
) -> Route:
    """The route and the red owners of one floor family (plan 5.8, reviews B3 and M3).

    ``cells`` maps ``(configuration, law, count)`` to its positive and control summary rows
    (each with ``cell``, ``passed`` and the floor readings). The floor is :func:`family_floor`
    of the per-count passes. A red below the floor is held under :func:`red_reading` (positive)
    or :data:`CONTROL_READING` (control). A red at or above the anchor makes the route a
    defect suspect; it is held under :data:`CHANCE_READING` only when ``diagnoses`` (default
    :data:`NO_DEFECT_DIAGNOSES`) records a no-defect diagnosis for it **and** it agrees with
    its committed counterpart in ``counterparts`` (default :func:`anchor_counterparts`, by
    :func:`agrees_with_committed`; review B3). Otherwise nobody holds it: a red that disagrees
    with the committed anchor is a measured change, and plan 5.8 (rev 3.7) withdraws the
    support of the shipped floor for that family. A red at or above the anchor with no committed
    counterpart at all (no cell of the committed study at that law and count) is never read as
    chance: the route is ``"redeclaration: {counts}"``, the floor stays at the anchor, and the
    family goes to a redeclaration.
    """
    diagnoses = NO_DEFECT_DIAGNOSES if diagnoses is None else diagnoses
    counterparts = anchor_counterparts() if counterparts is None else counterparts
    verdicts = {
        key: bool(positive["passed"]) and bool(control["passed"])
        for key, (positive, control) in cells.items()
    }
    configurations = sorted({key[0] for key in cells})
    passes = count_passes(verdicts, configurations)
    floor = family_floor(passes, anchor=anchor)
    high = sorted(count for count, ok in passes.items() if not ok and count >= anchor)
    uncompared = sorted(
        {
            count
            for (_, _, count), pair in cells.items()
            if count >= anchor
            for row in pair
            if not bool(row["passed"]) and str(row["cell"]) not in counterparts
        }
    )
    if uncompared:
        floor = anchor
        route = f"{REDECLARATION}: {', '.join(str(count) for count in uncompared)}"
    elif high:
        route = f"defect suspect: {', '.join(str(count) for count in high)}"
    elif floor < anchor:
        route = LOWERED
    else:
        route = f"kept: {max(c for c, ok in passes.items() if not ok)} fails"
    owners: dict[str, tuple[str, str] | None] = {}
    for (_, _, count), (positive, control) in sorted(cells.items()):
        for row, is_control in ((positive, False), (control, True)):
            if bool(row["passed"]):
                continue
            name = f"{FAMILY}/{row['cell']}"
            if count >= anchor or count >= floor:
                chance = name in diagnoses and agrees_with_committed(
                    row, counterparts.get(str(row["cell"])), control=is_control
                )
                owners[name] = (RED_OWNER, CHANCE_READING) if chance else None
            elif is_control:
                owners[name] = (RED_OWNER, CONTROL_READING)
            else:
                owners[name] = (RED_OWNER, red_reading(positive))
    return Route(family, floor, route, owners)


def candidate_owners(summary: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Every red ``few_cluster_candidate`` and pooled row, held under the candidate reading."""
    rows = summary.loc[summary["property"].isin([CANDIDATE_FAMILY, POOLED_FAMILY])]
    red = rows.loc[~rows["property_passed"].astype(bool)]
    return {
        f"{family}/{cell}": (RED_OWNER, CANDIDATE_READING)
        for family, cell in zip(red["property"], red["cell"], strict=True)
    }


@dataclasses.dataclass(frozen=True)
class Outcome:
    """The declared consequence for one compared or floor family (plan 5.7 and 5.8).

    Parameters
    ----------
    family : str
        The package's floor family (:data:`PACKAGE_FAMILIES`).
    floor : int
        The floor the package ships: :math:`F_f(A)` under the 5.8 rule, or :math:`F_f(C)` when
        (C) becomes the default (rule 6).
    route : str
        The :class:`Route` route.
    jackknife : str
        ``"default"``, ``"option"``, ``"refused: {reason}"``, or ``"not compared"``.
    wang_adopted : bool
        Whether (B) becomes the default reference (rule 3: it wins every compared family).
    """

    family: str
    floor: int
    route: str
    jackknife: str
    wang_adopted: bool


def outcomes(routes: Mapping[str, Route], decisions: Mapping[str, Decision]) -> dict[str, Outcome]:
    """Fold each family's route and part (c) decision into its declared outcome (M3).

    (B) is adopted only when it wins every compared family (rule 3). (C) becomes a family's
    default when it wins there (rule 4), and that family's floor is then its admissibility
    count (rule 6); an admissible (C) that does not win ships as an option; an inadmissible
    (C) is refused with the measured reason (rule 5).
    """
    wang = bool(decisions) and all(d.winner == WANG_ARM for d in decisions.values())
    out = {}
    for family, route in routes.items():
        decision = decisions.get(family)
        floor = route.floor
        if decision is None:
            status = "not compared"
        elif decision.admissible.get(JACKKNIFE_ARM) is None:
            status = f"refused: {decision.reasons.get(JACKKNIFE_ARM, 'not admissible')}"
        elif decision.winner == JACKKNIFE_ARM:
            status = "default"
            floor = int(decision.admissible[JACKKNIFE_ARM] or floor)
        else:
            status = "option"
        out[family] = Outcome(PACKAGE_FAMILIES[family], floor, route.route, status, wang)
    return out


def declared_outcome(
    summary_a: pd.DataFrame,
    rows_a: pd.DataFrame,
    summary_b: pd.DataFrame,
    rows_b: pd.DataFrame,
) -> dict[str, Outcome]:
    """The declared outcome of every floor family of S-a and S-b, from their summaries and rows.

    Rule 3 reads both studies' decisions together, so the two are folded at once.
    """
    from tests.studies import few_cluster_floor_ltmle_properties as sb
    from tests.studies import few_cluster_floor_tmle_properties as sa

    routes = {**sa.routes(summary_a), "ltmle_cross_fitted": sb.route(summary_b)}
    # A d5 red at or above the anchor has no committed counterpart: it sends tmle_stacked, whose
    # configurations the d5 fits are, to a redeclaration with its floor at the anchor.
    reported = sa.reported_route(summary_a)
    if reported.route.startswith(REDECLARATION):
        stacked = routes["tmle_stacked"]
        routes["tmle_stacked"] = dataclasses.replace(
            stacked, floor=sa.study.ANCHORS["tmle_stacked"], route=f"{reported.route} (d5)"
        )
    decisions = {
        **sa.decisions(summary_a, rows_a),
        "ltmle_cross_fitted": sb.decision(summary_b, rows_b),
    }
    return outcomes(routes, decisions)


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
    The clauses are treated as independent, which they are not: both read the positive's
    cluster SE, and an inflated cluster SE helps clause 1 while it hurts clause 2. The pass
    events are negatively dependent, so the product overstates the joint power. It feeds only
    the predicted-red list and the M5 plug-in, never a gate.
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

    # The requested draws, not the kept rows: a lost draw is a miss in the all-draw coverage,
    # so the hits and the pilot size both count it (review m2).
    requested = float(positive["requested_draws"])
    # Clause 2 reads the positive's SE-ratio interval on its shown draws (``shown_only``), so
    # its point is the shown-only SE ratio too; the shared ``se_ratio`` also reads withheld rows.
    shown = float(positive.get("shown_se_ratio", math.nan))
    return {
        "coverage": float(positive["all_draw_coverage"]),
        "hits": float(positive["all_draw_coverage"]) * requested,
        "standardized_bias": float(positive["standardized_bias"]),
        "ratio": float(control["paired_se_ratio"]),
        "ratio_sd": sd(control, "paired_se_ratio_ci_lower", "paired_se_ratio_ci_upper"),
        "se_ratio": shown if math.isfinite(shown) else float(positive["se_ratio"]),
        "se_ratio_sd": sd(positive, "se_ratio_ci_lower", "se_ratio_ci_upper"),
        "pilot_draws": requested,
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
