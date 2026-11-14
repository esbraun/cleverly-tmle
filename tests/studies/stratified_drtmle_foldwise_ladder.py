r"""The contraction property cells of ``stratified-drtmle-foldwise`` (Fix-6 unit S, ladder).

The four ``X8-drtmle-one-sided-bias`` cells of ``canonical-stratified-drtmle`` fit
``reduced_crossfit="pooled"`` and read the stratified paper law's ``treatment_correct`` regime
at n = 2,000.  The X83 sizing pilot put the fold-wise default's rows at 0.29 (marginal), 0.04
(``v0``), 0.30 (``v1``) and 0.18 SD (``v2``), the marginal and ``v1`` outside the 0.25 SD margin
on 200 replications.  The law is H1 per stratum, so its conditioning terms vanish and a red
one-correct row is a second-order term that has not decayed, or a defect.  These cells fit the
default on a size ladder and read the rate; they are the property cells of the gated record
``stratified-drtmle-foldwise`` (:mod:`tests.studies.stratified_drtmle_foldwise`), which names
them, and this module is its properties module.

**Design.**  :func:`tests.studies.canonical_stratified_drtmle.fit_cleverly` with
``reduced_crossfit="foldwise"``, at ``"treatment_correct"`` and at the ``"both_wrong"``
control, on rungs n = 2,000, 4,000 and 8,000 (smallest stratum about 400 to 1,600 rows).  Each
fit gives the marginal ATE and the three stratum ATEs.  Rows, in the family :data:`FAMILY`:
each rung's level cell ``<label>__<regime>_n<size>`` (bias, coverage, SE ratio) and each label's
rate cell ``<label>__rate_<regime>`` (log |bias| on log n, its 99% bootstrap interval).

**Pairing.**  The ``treatment_correct`` n = 2,000 rung draws
``stream_seed(canonical-stratified-drtmle, "property_sample", "treatment_correct", r)``, the X8
cells' stream, so replication ``r < 800`` is the X8 cells' registered replication ``r``, and
later replications continue that stream.  Every other rung draws a fresh named stream of the
record, whose resampling seed is fresh.

**Budgets, declared before any run** (the Fix-6 declaration, :data:`BUDGETS`).  6,000 at each
outer ``treatment_correct`` rung and 800 at the middle one, whose slope weight is zero; 400 per
control rung.  Each ``treatment_correct`` rung's level rows are read at
:data:`VERDICT_REPLICATES` (800, the X8 cells' budget); the extra outer replications serve the
slope only.  :func:`planned_outer_budget` sizes the outer rungs so that, when the bias is second
order, the predictive probability that the marginal's and ``v1``'s slope intervals both reach the
resolution target is at least 0.95; it reads 0.962 at 6,000.  Its planning values
(:func:`positive_plan`) are the inverse-variance pool of the pilot's n = 8,000 rung and half its
n = 2,000 rung (marginal 0.246, ``v1`` 0.145 SD), because a second-order bias halves in SD units
from 2,000 to 8,000; that is the hypothesis the reading tests, and no margin moves.
:func:`planned_control_budget` sizes the control so that it fails to contract with probability
at least 0.99; it reads 1.000 at 400.  The ladder costs about :data:`CPU_HOURS` CPU-hours at the
pilot's seconds per fit.  The rung rows were never in the sizing, so the rev 2.6 reading
below changes no budget.

**Reading, declared before the run** (:func:`slope_reading`, :func:`se_ratio_reading`,
:func:`x8_acceptance`).  The marginal and ``v1`` ``treatment_correct`` slopes are always read:

=================================================  ==================================
99% slope interval                                 reading
=================================================  ==================================
half-width above 0.25                              ``not resolved``
upper end at or above 0                            ``not shown to contract``
upper end below -1/2 (faster than -1 counts)       ``consistent with second order``
any other (contains -1/2, or upper end in          ``contracting or unresolved``
[-1/2, 0))
=================================================  ==================================

The escape hatch is read first: X8's acceptance names both rows and reads a half-width above
0.25 as ``not resolved`` whatever the interval's position.  ``v0`` and ``v2`` publish their
slopes at the width they reach and are ``not read``.  The ``both_wrong`` control's marginal and
``v1`` slopes must fail to contract (upper end at or above -1/2); a control that contracts is a
design failure.  Each ``treatment_correct`` level row is ``clean`` when its 99% SE-ratio
interval lies inside 0.8 to 1.2, and ``defect suspect`` otherwise (Fix-6 rev 2.6, finding B1).
Its coverage is published, not gated, beside :func:`implied_coverage`, the coverage its measured
standardized bias ``b`` and SE ratio ``r`` imply for a normal estimator,
``Phi(z r - b) - Phi(-z r - b)`` with ``z = Phi^-1(0.975)``: at ``r = 1`` a bias of 0.5 SD
alone gives 0.921.  A Wald interval around a biased estimate under-covers by exactly the bias
the slope reading already reads, so gating coverage again would fail the acceptance most of the
time under the second-order hypothesis it tests; a first-order term shows as SE
miscalibration, which the SE-ratio row gates.  The control's level rows are ``not read``.

**Verdicts.**  The record is gated.  A ``treatment_correct`` level row passes when it is
``clean``: it is judged on the SE ratio only, and its bias and coverage are published.  A read
label's ``treatment_correct`` rate row passes when it reads ``consistent with second order``, and
its control rate row when it fails to contract.  The ``not read`` rows pass by declaration.
Every red row of the ladder, a rung SE-ratio row included, is a defect suspect and is diagnosed;
no ladder row has a finite-sample owner.

The ``X8-drtmle-one-sided-bias`` acceptance, declared verbatim before any run, is Fix-6's D4
text as rev 2.1 left it, with the record named by its slug after the ladder became the property
cells of ``stratified-drtmle-foldwise``, and its rung clause as rev 2.6 words it
(:data:`X8_ACCEPTANCE`).  :func:`x8_acceptance` reads it: it holds when the marginal and ``v1``
read ``consistent with second order``, the control fails to contract, and no rung's SE-ratio
row is red.  Otherwise the red is a defect suspect and is diagnosed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_stratified_drtmle as canonical
from tests.studies import stratified_drtmle_foldwise as primary
from tests.studies.drtmle_foldwise_sizing import (
    LadderPlan,
    Pilot,
    committed_planning,
    control_budget,
    ladder_plan_from_rows,
    planned,
    predictive_outer_budget,
    second_order_pool,
)
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts, replication
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    ratio_intervals,
    replicate_row,
    require_accounted,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    finish,
    fitted_rate_row,
)
from tests.studies.evidence.registry import Margins
from tests.studies.evidence.seeds import stream_seed
from tests.studies.stratified_drtmle_foldwise import (
    CONTROL,
    FAMILY,
    LABELS,
    POSITIVE,
    READ_LABELS,
    REGIMES,
    SIZES,
    level_cell,
    rate_cell,
)

#: The rung whose stream is the X8 cells' stream.
PAIRED_RUNG = (POSITIVE, 2_000)
#: The X8 cells' registered budget, which the paired rung's first replications reproduce.
REGISTERED_REPLICATES = 800

# ------------------------------------------------------------------------------- the reading
#: The declared resolution target: the 99% slope interval's half-width for each read label.
RESOLUTION_TARGET = 0.25
#: The slope a first-order term predicts; an upper end below it reads second order.
FIRST_ORDER_SLOPE = -0.5

CONSISTENT = "consistent with second order"
CONTRACTING = "contracting or unresolved"
NOT_CONTRACTING = "not shown to contract"
NOT_RESOLVED = "not resolved"
NOT_READ = "not read"
FAILS_TO_CONTRACT = "fails to contract"
CONTROL_CONTRACTS = "contracts: design failure"
CLEAN = "clean"
DEFECT_SUSPECT = "defect suspect"

ACCEPTED = "X8 acceptance holds"
DEFECT = "defect suspect: diagnose"
DESIGN_FAILURE = "design failure: the control contracts"

#: The ``X8-drtmle-one-sided-bias`` acceptance, declared verbatim before any run.
X8_ACCEPTANCE = (
    'The four cells fit `reduced_crossfit="pooled"` and keep their verdicts. The owner closes '
    "as a finite-sample limit of the default when, in `stratified-drtmle-foldwise`, the "
    "99% slope interval of log |bias| on log n has its upper end below -1/2 for the marginal "
    "ATE and for `v1` (each read whatever its realised half-width; a half-width above 0.25 for "
    "either reads `not resolved`, and the owner does not close), the `both_wrong` control "
    "fails to contract, and no rung's SE-ratio row is red (each rung's coverage is published "
    "beside the coverage its measured bias and SE ratio imply). Otherwise the red is a defect "
    "suspect and is diagnosed, and this owner does not close."
)

#: The columns this family adds to the summary.
EXTRA_COLUMNS = ("reading", "half_width", "implied_coverage")


def slope_reading(interval: Interval) -> str:
    """The reading of a read label's ``treatment_correct`` slope interval (module docstring).

    Parameters
    ----------
    interval : Interval
        The 99% bootstrap interval of the log |bias| on log n slope.

    Returns
    -------
    str
        :data:`NOT_RESOLVED`, :data:`NOT_CONTRACTING`, :data:`CONSISTENT` or
        :data:`CONTRACTING`.
    """
    half_width = (interval.high - interval.low) / 2
    if not half_width <= RESOLUTION_TARGET:
        return NOT_RESOLVED
    if interval.high >= 0.0:
        return NOT_CONTRACTING
    if interval.high < FIRST_ORDER_SLOPE:
        return CONSISTENT
    return CONTRACTING


def control_reading(interval: Interval) -> str:
    """The reading of a read label's ``both_wrong`` slope interval.

    Parameters
    ----------
    interval : Interval
        The 99% bootstrap interval of the control's slope.

    Returns
    -------
    str
        :data:`FAILS_TO_CONTRACT` when the upper end is at or above -1/2, else
        :data:`CONTROL_CONTRACTS`.
    """
    return FAILS_TO_CONTRACT if interval.high >= FIRST_ORDER_SLOPE else CONTROL_CONTRACTS


def se_ratio_reading(se_ratio: Interval, margins: Margins) -> str:
    """The reading of a ``treatment_correct`` level row, on its SE ratio alone.

    Parameters
    ----------
    se_ratio : Interval
        The 99% bootstrap interval of the mean reported SE over the empirical SD.
    margins : Margins
        The SE-ratio sanity band.

    Returns
    -------
    str
        :data:`CLEAN` when the interval lies inside the band, else :data:`DEFECT_SUSPECT`.
    """
    return CLEAN if se_ratio.within(*margins.se_ratio_sanity) else DEFECT_SUSPECT


def implied_coverage(standardized_bias: float, se_ratio: float, level: float = 0.95) -> float:
    """The coverage a normal estimator's measured bias and SE ratio imply for its Wald interval.

    With error ``N(b * sd, sd^2)`` and reported SE ``r * sd``, the interval
    ``estimate +/- z * SE`` covers with probability ``Phi(z r - b) - Phi(-z r - b)``.

    Parameters
    ----------
    standardized_bias : float
        ``b``, the mean error over the empirical SD.
    se_ratio : float
        ``r``, the mean reported SE over the empirical SD.
    level : float
        The interval's nominal level.

    Returns
    -------
    float
        The implied coverage.
    """
    z = float(norm.ppf(0.5 + level / 2))
    return float(
        norm.cdf(z * se_ratio - standardized_bias) - norm.cdf(-z * se_ratio - standardized_bias)
    )


# --------------------------------------------------------------------------------- the sizes
#: The declared budgets, ``(n = 2,000, 4,000, 8,000)`` per regime: :func:`planned_outer_budget`
#: gives the outer ``treatment_correct`` rungs (predictive pass 0.962) and
#: :func:`planned_control_budget` the control (predictive pass 1.000).
BUDGETS: Mapping[str, tuple[int, int, int]] = {
    POSITIVE: (6_000, 800, 6_000),
    CONTROL: (400, 400, 400),
}
#: The replications each ``treatment_correct`` rung's level rows are read at.
VERDICT_REPLICATES = 800

#: The seed of the ladder's planning draws.
PLANNING_SEED = 2026111224
#: The predictive probability the outer rungs must resolve every read label with.
RESOLUTION_PROBABILITY = 0.95
#: The predictive probability the control must fail to contract with.
CONTROL_PROBABILITY = 0.99
#: The pilot groups of the ``treatment_correct`` rungs: the top rung's and the lower rung's
#: whose values :func:`positive_plan` carries up as second order.
PLANNING_GROUPS = ("ladder_tc_8000", "ladder_tc_2000")
#: The ladder's cost in CPU-hours at the pilot's seconds per fit
#: (:data:`~tests.studies.drtmle_foldwise_sizing.PLANNING_PATH`).
CPU_HOURS = 54.7


def positive_plan(planning: Mapping[str, Pilot | LadderPlan] | None = None) -> LadderPlan:
    """The read labels' planning values at the top rung, under the second-order hypothesis.

    The inverse-variance pool of the n = 8,000 rung's standardized biases and half the
    n = 2,000 rung's (:func:`~tests.studies.drtmle_foldwise_sizing.second_order_pool`), with the
    top rung's error correlation.  A second-order bias halves in SD units from n = 2,000 to
    8,000, which is the hypothesis the slope reading tests.

    Parameters
    ----------
    planning : Mapping or None
        Pilot group to its summary; the committed summaries when ``None``.

    Returns
    -------
    LadderPlan
        The plan at n = 8,000.
    """
    top, low = (planned(planning, group, LadderPlan) for group in PLANNING_GROUPS)
    return second_order_pool(top, low)


@cache
def control_plan() -> LadderPlan:
    """The control's planning values: the registered pooled ``both_wrong`` cells at n = 2,000.

    ``canonical-stratified-drtmle``'s ``X8`` ``both_wrong`` cells of the read labels (R = 800,
    ``reduced_crossfit="pooled"``).  The control's bias is first order, which the reduced
    regressions' fold policy does not remove, so a pooled cell plans the fold-wise control.

    Returns
    -------
    LadderPlan
        The plan, with the labels' per-replication error correlation.
    """
    rows = pd.read_parquet(canonical.STUDY.artifact("property-replicates.parquet"))
    cells = {f"{label}__{CONTROL}": label for label in READ_LABELS}
    own = rows.loc[rows["cell"].isin(cells)].assign(estimand=lambda f: f["cell"].map(cells))
    return ladder_plan_from_rows(
        own,
        source="canonical-stratified-drtmle property-replicates.parquet, both_wrong, pooled",
        labels={label: label for label in READ_LABELS},
        size=2_000,
    )


def planned_outer_budget(
    planning: Mapping[str, Pilot | LadderPlan] | None = None,
) -> tuple[int, float, bool]:
    """The outer ``treatment_correct`` budget that resolves every read label (module docstring).

    Parameters
    ----------
    planning : Mapping or None
        Pilot group to its summary, read by :func:`positive_plan`; the committed summaries
        when ``None``.

    Returns
    -------
    tuple
        ``(R, predictive pass, sized)`` on a 400 grid from the verdict budget to 8,000, the
        middle rung at :data:`VERDICT_REPLICATES`.
    """
    return predictive_outer_budget(
        positive_plan(planning),
        SIZES,
        VERDICT_REPLICATES,
        target=RESOLUTION_TARGET,
        level=Margins().confidence_level,
        probability=RESOLUTION_PROBABILITY,
        floor=VERDICT_REPLICATES,
        seed=PLANNING_SEED,
    )


def planned_control_budget(plan: LadderPlan | None = None) -> tuple[int, float, bool]:
    """The per-rung ``both_wrong`` budget at which the control fails to contract.

    Parameters
    ----------
    plan : LadderPlan or None
        The control's planning values; :func:`control_plan` when ``None``.

    Returns
    -------
    tuple
        ``(R, predictive pass, sized)`` on a 400 grid from 400 to 8,000.
    """
    return control_budget(
        control_plan() if plan is None else plan,
        SIZES,
        first_order=FIRST_ORDER_SLOPE,
        level=Margins().confidence_level,
        probability=CONTROL_PROBABILITY,
        seed=PLANNING_SEED,
    )


def cpu_hours(seconds: Mapping[str, float] | None = None) -> float:
    """The ladder's cost in CPU-hours at the pilot's mean seconds per fit.

    The pilot timed ``treatment_correct`` fits at n = 2,000 and 8,000; a fit at 4,000 is
    costed at their mean, and a ``both_wrong`` fit as a ``treatment_correct`` fit of its size.

    Parameters
    ----------
    seconds : Mapping or None
        Pilot group to its seconds per fit; the committed values when ``None``.

    Returns
    -------
    float
        The CPU-hours of :data:`BUDGETS`.
    """
    per_fit = committed_planning()[1] if seconds is None else seconds
    low, top = per_fit["ladder_tc_2000"], per_fit["ladder_tc_8000"]
    by_size = (low, (low + top) / 2, top)
    total = sum(
        budget * cost
        for regime in REGIMES
        for budget, cost in zip(BUDGETS[regime], by_size, strict=True)
    )
    return total / 3_600


# ----------------------------------------------------------------------------- the ladder
def sample_seed(regime: str, size: int, replicate: int) -> int:
    """The sample seed of one replication of one rung.

    The paired rung reads the X8 cells' stream; every other rung reads a fresh stream of this
    record named by its regime and size.
    """
    if (regime, size) == PAIRED_RUNG:
        return stream_seed(canonical.STUDY, "property_sample", POSITIVE, replicate)
    return stream_seed(primary.STUDY, "property_sample", f"{regime}_n{size}", replicate)


_Key = tuple[tuple[str, int, int, int]]
_Drawn = tuple[pd.DataFrame, dict[str, float]]


def _draw_rung(key: _Key) -> _Drawn:
    """The sample a key fits, a function of the key alone."""
    regime, size, replicate, _ = key[0]
    return canonical.draw_from_seed(canonical.SCENARIO, size, sample_seed(regime, size, replicate))


def _cells_of(key: _Key) -> list[str]:
    regime, size, _, _ = key[0]
    return [f"{FAMILY}/{level_cell(label, regime, size)}" for label in LABELS]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[0][2]


def _seed_of(key: _Key) -> int:
    regime, size, replicate, _ = key[0]
    return sample_seed(regime, size, replicate)


def _fit_key(key: _Key, drawn: _Drawn) -> list[dict[str, Any]]:
    regime, size, replicate, requested = key[0]
    frame, truth = drawn
    result = primary.fit_cleverly(frame, regime)
    return [
        replicate_row(
            property_name=FAMILY,
            cell=level_cell(label, regime, size),
            role="control" if regime == CONTROL else "positive",
            replicate=replicate,
            n=size,
            requested=requested,
            truth=float(truth[canonical.ate_name(label)]),
            estimate=result[canonical.ate_name(label)],
            alpha=primary.STUDY.margins.alpha,
        )
        for label in LABELS
    ]


def payloads(budget: int | None = None, sizes: Sequence[int] = SIZES) -> list[_Key]:
    """Every ladder replication; ``budget`` caps each rung and ``sizes`` picks rungs (smoke)."""
    out: list[_Key] = []
    for regime in REGIMES:
        for index, size in enumerate(SIZES):
            if size not in sizes:
                continue
            declared = BUDGETS[regime][index]
            count = declared if budget is None else min(budget, declared)
            out += [((regime, size, r, count),) for r in range(count)]
    return out


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: before the record's first run.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, sizes: Sequence[int] = SIZES
) -> pd.DataFrame:
    """Fit every ladder replication; ``budget`` and ``sizes`` cut a smoke run.

    A declared run (``budget=None``) passes the record's guard first.
    """
    if budget is None:
        primary.assert_declared()
    outcomes = replication.fit_properties(
        payloads(budget, sizes),
        _fit_key,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw_rung,
        seed_of=_seed_of,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def at_verdict_budget(
    rows: pd.DataFrame, verdict_replicates: int = VERDICT_REPLICATES
) -> pd.DataFrame:
    """Each ``treatment_correct`` rung's rows below the budget its level rows are read at.

    A rung keeps the ids below the budget and never refills a lost one from a later id, as
    :func:`tests.studies.evidence.property_verdicts.summarize_contraction_properties` does.
    The control rungs and the other families are returned untouched.

    Parameters
    ----------
    rows : pandas.DataFrame
        Every replication row of the record.
    verdict_replicates : int
        The budget.

    Returns
    -------
    pandas.DataFrame
        The rows the level verdicts read.
    """
    rung = (rows["property"] == FAMILY) & (rows["role"] == "positive")
    if not rung.any():
        return rows
    require_accounted(rows.loc[rung])
    kept = []
    for cell, group in rows.loc[rung].groupby("cell", sort=True):
        requested = int(group["requested_replicates"].iloc[0])
        if requested < verdict_replicates:
            raise ValueError(
                f"{FAMILY}/{cell} requested {requested} replications, fewer than the "
                f"{verdict_replicates} its level rows are read at"
            )
        part = group.loc[group["replicate"] < verdict_replicates].sort_values("replicate").copy()
        part["requested_replicates"] = verdict_replicates
        part["failed_replicates"] = verdict_replicates - len(part)
        kept.append(part)
    return pd.concat([rows.loc[~rung], *kept], ignore_index=True)


def _level_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Write each level row's SE-ratio interval, reading and verdict, in place."""
    margins = primary.STUDY.margins
    for index in summary.index[summary["property"] == FAMILY]:
        row = summary.loc[index]
        cell = str(row["cell"])
        group = rows.loc[(rows["property"] == FAMILY) & (rows["cell"] == cell)]
        ratio = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(primary.STUDY, FAMILY, cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = ratio.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio.high
        summary.loc[index, "implied_coverage"] = implied_coverage(
            float(row["standardized_bias"]), float(row["se_ratio"]), 1.0 - margins.alpha
        )
        if row["role"] == "control":
            summary.loc[index, "reading"] = NOT_READ
            summary.loc[index, "passed"] = True
            continue
        reading = se_ratio_reading(ratio, margins)
        summary.loc[index, "reading"] = reading
        summary.loc[index, "passed"] = reading == CLEAN


def _rate_rows(rows: pd.DataFrame, columns: Any) -> list[dict[str, Any]]:
    """Each label's fitted slope per regime, with its reading and verdict."""
    ladder = rows.loc[rows["property"] == FAMILY]
    out = []
    for regime in REGIMES:
        for label in LABELS:
            cells = [level_cell(label, regime, size) for size in SIZES]
            row = fitted_rate_row(
                ladder.loc[ladder["cell"].isin(cells)],
                primary.STUDY,
                columns,
                ladder_property=FAMILY,
                property_name=FAMILY,
                cell=rate_cell(label, regime),
                role="control" if regime == CONTROL else "positive",
                statistic="bias",
                seed_labels=(FAMILY, label, regime),
                verdict=lambda fitted: True,
            )
            interval = Interval(float(row["slope_ci_lower"]), float(row["slope_ci_upper"]))
            row["half_width"] = (interval.high - interval.low) / 2
            if label not in READ_LABELS:
                reading = NOT_READ
            elif regime == CONTROL:
                reading = control_reading(interval)
            else:
                reading = slope_reading(interval)
            row["reading"] = reading
            row["passed"] = reading in (NOT_READ, CONSISTENT, FAILS_TO_CONTRACT)
            out.append(row)
    return out


def summarize_properties(
    rows: pd.DataFrame, *, verdict_replicates: int = VERDICT_REPLICATES
) -> pd.DataFrame:
    """The level rows at their verdict budget, then every slope on the full rows.

    Parameters
    ----------
    rows : pandas.DataFrame
        Every replication row of the record.
    verdict_replicates : int
        The budget the level rows are read at; a smoke run passes its own.

    Returns
    -------
    pandas.DataFrame
        The published summary.
    """
    levels = at_verdict_budget(rows, verdict_replicates)
    summary, rates = apply_shared_verdicts(
        levels, primary.STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=()
    )
    summary["reading"] = summary["reading"].astype(object)
    _level_verdicts(summary, levels)
    rates.extend(_rate_rows(rows, summary.columns))
    return finish(summary, rates)


def x8_acceptance(summary: pd.DataFrame) -> str:
    """The outcome the declared reading gives the ``X8`` owner (module docstring).

    Parameters
    ----------
    summary : pandas.DataFrame
        The record's published summary.

    Returns
    -------
    str
        :data:`DESIGN_FAILURE` when a read control slope contracts; else :data:`ACCEPTED` when
        every read ``treatment_correct`` slope is consistent with second order and no rung's
        SE-ratio row is red; else :data:`DEFECT`.
    """
    own = summary.loc[summary["property"] == FAMILY].set_index("cell")
    controls = [own.loc[rate_cell(label, CONTROL), "reading"] for label in READ_LABELS]
    if any(reading != FAILS_TO_CONTRACT for reading in controls):
        return DESIGN_FAILURE
    slopes = [own.loc[rate_cell(label, POSITIVE), "reading"] for label in READ_LABELS]
    levels = own.loc[[level_cell(label, POSITIVE, n) for label in LABELS for n in SIZES], "reading"]
    if all(reading == CONSISTENT for reading in slopes) and not (levels == DEFECT_SUSPECT).any():
        return ACCEPTED
    return DEFECT


# ------------------------------------------------------------------------- failure accounting
def _is_rate(ctx: pf.Context, row: pd.Series) -> bool:
    return "__rate_" in str(row["cell"])


def _label(row: pd.Series) -> str:
    return str(row["cell"]).split("__", 1)[0]


def _ladder(row: pd.Series) -> tuple[str, str, str, tuple[str, ...]]:
    """The rungs, statistic and seed labels of a rate row (:func:`_rate_rows`)."""
    label, rest = str(row["cell"]).split("__", 1)
    regime = rest.removeprefix("rate_")
    return FAMILY, f"{label}__{regime}_n", "bias", (FAMILY, label, regime)


def _second_order(ctx: pf.Context, row: pd.Series, iv: Interval) -> pf.Judged:
    slack = min(RESOLUTION_TARGET - (iv.high - iv.low) / 2, FIRST_ORDER_SLOPE - iv.high)
    return pf.Judged(slack, slope_reading(iv) == CONSISTENT)


def _fails_to_contract(ctx: pf.Context, row: pd.Series, iv: Interval) -> pf.Judged:
    return pf.Judged(iv.high - FIRST_ORDER_SLOPE, control_reading(iv) == FAILS_TO_CONTRACT)


def _at_budget(clause: pf.Clause) -> pf.Clause:
    """``clause`` with its candidates read on the rows its level verdict reads.

    A level row reads its rung at a verdict budget (:data:`VERDICT_REPLICATES` in a declared run,
    :func:`at_verdict_budget`), while the outer rungs hold more replications for the slopes.  The
    row's own replicates and losses give that budget.  The candidates must read the same
    replications, or at no loss they would not reproduce the complete case (F18).
    """
    inner = clause.candidates
    assert inner is not None

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        budget = int(row["replicates"]) + ctx.k(row)
        return inner(dataclasses.replace(ctx, rows=at_verdict_budget(ctx.rows, budget)), row)

    return dataclasses.replace(clause, candidates=candidates)


def _rule() -> pf.FamilyRule:
    def level(ctx: pf.Context, row: pd.Series) -> bool:
        return str(row["role"]) == "positive" and not _is_rate(ctx, row)

    def read_rate(role: str) -> pf.Applies:
        return lambda ctx, row: (
            str(row["role"]) == role and _is_rate(ctx, row) and _label(row) in READ_LABELS
        )

    return pf.FamilyRule(
        families=frozenset({FAMILY}),
        shared="summarize_properties",
        clauses=(
            _at_budget(
                pf.se_interval_clause(
                    "rung se ratio in sanity band",
                    which="se_ratio",
                    judge=pf.band_judge(lambda ctx: ctx.margins.se_ratio_sanity),
                    seed=lambda ctx, row: stream_seed(ctx.record, FAMILY, str(row["cell"])),
                    applies=level,
                )
            ),
            pf.rate_slope_clause(
                "slope consistent with second order",
                judge=_second_order,
                ladder=_ladder,
                applies=read_rate("positive"),
            ),
            pf.rate_slope_clause(
                "control slope fails to contract",
                judge=_fails_to_contract,
                ladder=_ladder,
                applies=read_rate("control"),
            ),
        ),
    )


#: The failure accounting's declared rule for :data:`FAMILY`.
PROPERTY_RULES = (_rule(),)
