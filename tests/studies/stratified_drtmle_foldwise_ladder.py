r"""Fix-6 unit S, ladder: the fold-wise default's stratified ``treatment_correct`` bias rate.

The four ``X8-drtmle-one-sided-bias`` cells of ``canonical-stratified-drtmle`` fit
``reduced_crossfit="pooled"`` and read the stratified paper law's ``treatment_correct`` regime
at n = 2,000.  The X83 sizing pilot put the fold-wise default's rows at 0.29 (marginal), 0.04
(``v0``), 0.30 (``v1``) and 0.18 SD (``v2``), the marginal and ``v1`` outside the 0.25 SD margin
on 200 replications.  The law is H1 per stratum, so its conditioning terms vanish and a red
one-correct row is a second-order term that has not decayed, or a defect.  This record fits the
default on a size ladder and reads the rate.

**Design.**  :func:`tests.studies.canonical_stratified_drtmle.fit_cleverly` with
``reduced_crossfit="foldwise"``, at ``"treatment_correct"`` and at the ``"both_wrong"``
control, on rungs n = 2,000, 4,000 and 8,000 (smallest stratum about 400 to 1,600 rows).  Each
fit gives the marginal ATE and the three stratum ATEs.  Rows, in the family :data:`FAMILY`:
each rung's level cell ``<label>__<regime>_n<size>`` (bias, coverage, SE ratio) and each label's
rate cell ``<label>__rate_<regime>`` (log |bias| on log n, its 99% bootstrap interval).

**Pairing.**  The ``treatment_correct`` n = 2,000 rung draws
``stream_seed(canonical-stratified-drtmle, "property_sample", "treatment_correct", r)``, the X8
cells' stream, so replication ``r < 800`` is the X8 cells' registered replication ``r``, and
later replications continue that stream.  Every other rung draws a fresh named stream of this
record.  :data:`RESAMPLING_SEED` is fresh.

**Budgets.**  Planning values (:data:`PLANNING_BUDGETS`): 3,300 at each outer
``treatment_correct`` rung and 800 at the middle one, whose slope weight is zero; 400 per
control rung.  Each ``treatment_correct`` rung's level rows are read at
:data:`VERDICT_REPLICATES` (800, the X8 cells' budget); the extra outer replications serve the
slope only.  :func:`planned_outer_budget` recomputes the outer budget from the planning
standardized biases and the resolution target; the sizing pilot replaces the planning values.

**Reading, declared before the run** (:func:`slope_reading`, :func:`calibration_reading`,
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
design failure.  Each ``treatment_correct`` level row is ``clean`` when its exact coverage
interval clears the 0.90 floor and its SE-ratio interval lies inside 0.8 to 1.2,
``finite-sample`` when it is red but its SE-ratio interval contains 1 and its coverage interval
contains 0.95, and ``defect suspect`` otherwise.  The control's level rows are ``not read``.

The ``X8`` acceptance holds when the marginal and ``v1`` read ``consistent with second order``,
the control fails to contract, and no level row is a ``defect suspect``.  Otherwise the red is
a defect suspect and is diagnosed.  ``publication_policy="reporting"``.

**The primary phase is open.**  The registry and the regeneration driver require one, and the
Fix-6 plan names none for this record: its rows are property cells on the X8 stream, which a
primary phase cannot draw.  :data:`PRIMARY_SCENARIO` is a placeholder: the
``treatment_correct`` fit at n = 2,000 on this record's own fresh seed, reported rows that pass
by declaration.  :func:`assert_declared` refuses a declared run while
:data:`PRIMARY_DESIGN_OPEN` is set; the design review rules on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_stratified_drtmle as canonical
from tests.studies import drtmle_foldwise as foldwise
from tests.studies import stratified_drtmle_foldwise as primary
from tests.studies.drtmle_foldwise_sizing import outer_rung_budget
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
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS as PRIMARY_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed
from tests.studies.point_study_helpers import primary_rows

IMPLEMENTATION = foldwise.IMPLEMENTATION
#: The placeholder primary phase's own seed, fresh: no registered study draws from it.
SEED = 2026111204
#: Fresh entropy for every bootstrap and every fresh rung stream of this record.
RESAMPLING_SEED = 2026111214
PRIMARY_N = 2_000
PRIMARY_SCENARIO = "stratified_treatment_correct"
ESTIMANDS = primary.ESTIMANDS
FOLDWISE = foldwise.FOLDWISE

#: Set while the plan names no primary phase for this record (module docstring).
PRIMARY_DESIGN_OPEN = True

FAMILY = "stratified_contraction"
SIZES = (2_000, 4_000, 8_000)
POSITIVE = "treatment_correct"
CONTROL = "both_wrong"
REGIMES = (POSITIVE, CONTROL)
LABELS = canonical.ATE_LABELS
#: The labels the reading always reads; the X8 acceptance names both.
READ_LABELS = ("marginal_ate", "v1_ate")
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
FINITE_SAMPLE = "finite-sample"
DEFECT_SUSPECT = "defect suspect"

ACCEPTED = "X8 acceptance holds"
DEFECT = "defect suspect: diagnose"
DESIGN_FAILURE = "design failure: the control contracts"

#: The columns this family adds to the summary.
EXTRA_COLUMNS = ("reading", "half_width")


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


def calibration_reading(coverage: Interval, se_ratio: Interval, margins: Margins) -> str:
    """The reading of a ``treatment_correct`` level row's coverage and SE ratio.

    Parameters
    ----------
    coverage : Interval
        The 99% exact interval of the 95% Wald coverage.
    se_ratio : Interval
        The 99% bootstrap interval of the mean reported SE over the empirical SD.
    margins : Margins
        The coverage floor, the SE-ratio sanity band and the nominal level.

    Returns
    -------
    str
        :data:`CLEAN` when the coverage clears the floor and the SE ratio lies inside the band;
        else :data:`FINITE_SAMPLE` when the SE-ratio interval contains 1 and the coverage
        interval contains the nominal level (the finite-sample rule); else
        :data:`DEFECT_SUSPECT`.
    """
    if coverage.low >= margins.coverage_floor and se_ratio.within(*margins.se_ratio_sanity):
        return CLEAN
    if se_ratio.contains(1.0) and coverage.contains(1.0 - margins.alpha):
        return FINITE_SAMPLE
    return DEFECT_SUSPECT


# --------------------------------------------------------------------------------- the sizes
#: Planning budgets, not declared ones: ``(n = 2,000, 4,000, 8,000)`` per regime.
PLANNING_BUDGETS: Mapping[str, tuple[int, int, int]] = {
    POSITIVE: (3_300, 800, 3_300),
    CONTROL: (400, 400, 400),
}
#: ``False`` until the declaration commit sets the budgets from the sizing pilot.
BUDGETS_DECLARED = False
BUDGETS: Mapping[str, tuple[int, int, int]] = PLANNING_BUDGETS
#: The replications each ``treatment_correct`` rung's level rows are read at.
VERDICT_REPLICATES = 800
#: The placeholder primary phase's planning budget, the primaries' floor.
PRIMARY_REPLICATES = 1_000

#: The planning standardized biases of the read labels at each rung, before the sizing pilot:
#: the X83 pilot's fold-wise n = 2,000 rows (``pilot_b.csv``, 200 replications), carried up the
#: ladder by the second-order hypothesis the slope must resolve (``z`` halves per fourfold n).
#: The pilot measures n = 2,000 and n = 8,000 directly and replaces them.
PLANNING_STANDARDIZED_BIAS: Mapping[str, tuple[float, float, float]] = {
    label: (z, z / np.sqrt(2.0), z / 2.0)
    for label, z in (("marginal_ate", 0.2905), ("v1_ate", 0.3035))
}


def planned_outer_budget(
    standardized: Mapping[str, Sequence[float]] = PLANNING_STANDARDIZED_BIAS,
) -> int | None:
    """The outer-rung budget that brings every read label's projected half-width to target.

    Parameters
    ----------
    standardized : Mapping
        Label to its planning standardized bias at each rung.

    Returns
    -------
    int or None
        The largest of the labels' budgets on a 100 grid, or ``None`` when one label reaches
        the target at no budget up to 20,000.
    """
    budgets = []
    for label in READ_LABELS:
        budget = outer_rung_budget(
            standardized[label],
            SIZES,
            BUDGETS[POSITIVE][1],
            target=RESOLUTION_TARGET,
            level=Margins().confidence_level,
        )
        if budget is None:
            return None
        budgets.append(budget)
    return max(budgets)


def assert_declared() -> None:
    """Refuse a declared run while the design or the budgets are open.

    Raises
    ------
    RuntimeError
        While :data:`PRIMARY_DESIGN_OPEN` is set or :data:`BUDGETS_DECLARED` is ``False``.
    """
    if PRIMARY_DESIGN_OPEN:
        raise RuntimeError(
            "stratified-drtmle-foldwise-ladder has a placeholder primary phase; the design "
            "review rules on it before any declared run"
        )
    if not BUDGETS_DECLARED:
        raise RuntimeError(
            "stratified-drtmle-foldwise-ladder budgets are planning values; the sizing pilot "
            "and the declaration commit set them before any declared run"
        )


# ------------------------------------------------------------------------- the registration
def level_cell(label: str, regime: str, size: int) -> str:
    """The level cell of one label, regime and rung."""
    return f"{label}__{regime}_n{size}"


def rate_cell(label: str, regime: str) -> str:
    """The rate cell of one label and regime."""
    return f"{label}__rate_{regime}"


PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    FAMILY: tuple(
        sorted(
            [
                level_cell(label, regime, size)
                for label in LABELS
                for regime in REGIMES
                for size in SIZES
            ]
            + [rate_cell(label, regime) for label in LABELS for regime in REGIMES]
        )
    )
}

STUDY = StudyRecord(
    name="stratified DR-TMLE fold-wise one-correct ladder",
    slug="stratified-drtmle-foldwise-ladder",
    artifacts=ROOT / "tests" / "canonical" / "stratified_drtmle_foldwise_ladder",
    document="docs/technical-reference/method-evidence/stratified-dr-tmle-foldwise-ladder.md",
    anchor="stratified-dr-tmle-fold-wise-one-correct-ladder",
    scenarios={PRIMARY_SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: 0.01},
    reported_scenarios=frozenset({PRIMARY_SCENARIO}),
    modules=(
        "tests/studies/stratified_drtmle_foldwise_ladder.py",
        "tests/studies/stratified_drtmle_foldwise.py",
        "tests/studies/drtmle_foldwise_sizing.py",
        "tests/studies/canonical_stratified_drtmle.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/discrete_law.py",
        "tests/studies/drtmle_foldwise.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_failures.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/canonical_multi_arm_drtmle.py",
        "tests/studies/drtmle_foldwise_properties.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/multi_arm_drtmle_properties.py",
        "tests/studies/multi_arm_properties.py",
    ),
    runner_module="tests.studies.stratified_drtmle_foldwise_ladder",
    properties_module="tests.studies.stratified_drtmle_foldwise_ladder",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION: dict[str, Any] = {
    **primary.CONFIGURATION,
    "draws": (
        "the treatment_correct n = 2,000 rung draws canonical-stratified-drtmle's X8 stream "
        "(property_sample, treatment_correct), so replication r < 800 is the X8 cells' "
        "replication r; every other rung draws a fresh named stream of this record"
    ),
    "rungs": list(SIZES),
    "resolution_target": RESOLUTION_TARGET,
    "verdict_replicates": VERDICT_REPLICATES,
    "primary_design": "open: placeholder reported rows, pending the design review",
}


# --------------------------------------------------------------------- the primary phase
def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the stratified paper law with its supplied fold vector."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return canonical.draw_from_seed(canonical.SCENARIO, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this record's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return primary_rows(
        result=primary.fit_cleverly(frame, POSITIVE),
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every primary replication; a declared budget passes the guard first."""
    if replicates >= PRIMARY_REPLICATES:
        assert_declared()
    outcomes = replication.fit_primary(
        [(PRIMARY_SCENARIO, r, n) for r in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(PRIMARY_COLUMNS)])


# ----------------------------------------------------------------------------- the ladder
def sample_seed(regime: str, size: int, replicate: int) -> int:
    """The sample seed of one replication of one rung.

    The paired rung reads the X8 cells' stream; every other rung reads a fresh stream of this
    record named by its regime and size.
    """
    if (regime, size) == PAIRED_RUNG:
        return stream_seed(canonical.STUDY, "property_sample", POSITIVE, replicate)
    return stream_seed(STUDY, "property_sample", f"{regime}_n{size}", replicate)


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
            alpha=STUDY.margins.alpha,
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

    A declared run (``budget=None``) passes :func:`assert_declared` first.
    """
    if budget is None:
        assert_declared()
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
    margins = STUDY.margins
    for index in summary.index[summary["property"] == FAMILY]:
        row = summary.loc[index]
        cell = str(row["cell"])
        group = rows.loc[(rows["property"] == FAMILY) & (rows["cell"] == cell)]
        ratio = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, FAMILY, cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = ratio.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio.high
        if row["role"] == "control":
            summary.loc[index, "reading"] = NOT_READ
            summary.loc[index, "passed"] = True
            continue
        coverage = Interval(float(row["coverage_ci_lower"]), float(row["coverage_ci_upper"]))
        reading = calibration_reading(coverage, ratio, margins)
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
                STUDY,
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
        levels, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=()
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
        every read ``treatment_correct`` slope is consistent with second order and no level
        row is a defect suspect; else :data:`DEFECT`.
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
            pf.proportion_clause(
                "rung coverage floor",
                statistic="S2",
                judge=pf.floor_judge(lambda ctx: ctx.margins.coverage_floor),
                applies=level,
            ),
            pf.se_interval_clause(
                "rung se ratio in sanity band",
                which="se_ratio",
                judge=pf.band_judge(lambda ctx: ctx.margins.se_ratio_sanity),
                seed=lambda ctx, row: stream_seed(ctx.record, FAMILY, str(row["cell"])),
                applies=level,
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
