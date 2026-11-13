r"""Fix-6 unit S, primary: the fold-wise DR-TMLE default on the stratified primary.

The subject is ``canonical-stratified-drtmle``'s registered fit,
:func:`tests.studies.canonical_stratified_drtmle.fit_cleverly` at ``"both_correct"``, with
``reduced_crossfit="foldwise"``, the shipped default.  Everything else is that study's primary
configuration: the stratified paper law (``V`` in {0, 1, 2}), n = 2,000, its GLM learners, ten
supplied unstratified folds, every reduced regression fitted inside each stratum on
``(g_n, V)``.  The record is the truth, coverage and SE-ratio evidence a support row for the
default cross-fitted stratified fit needs (Fix-6 D6).

**Rows.**  The nine stratum names of ``canonical-stratified-drtmle`` and, as a new truth row,
the marginal ATE ``ate``.  The canonical study excludes the marginal because it has no R pair
(a marginal ``drtmle`` call reduces on ``g_n`` alone); this record has no comparator, and the
marginal's exact truth is the law's mixture.

**Pairing.**  :data:`SEED` **is** ``canonical-stratified-drtmle``'s seed and the scenario is its
scenario, so replication ``r < 800`` is that study's registered replication ``r``: the same rows
and the same fold vector.  Replications 800 and above continue the stream.
:data:`RESAMPLING_SEED` is fresh.

**The contraction property cells.**  The fold-wise default's stratified ``treatment_correct``
regime on a size ladder, with the ``both_wrong`` control: :data:`FAMILY`, fitted and judged by
:mod:`tests.studies.stratified_drtmle_foldwise_ladder` (the record's properties module, which
holds the design, the pairing with the X8 cells' stream and the declared reading).  Each rung's
level row is judged on coverage and SE ratio only, its bias published; the marginal ATE's and
``v1``'s rate rows are gated by the declared slope reading, and the control's by its failure to
contract.

**No comparator.**  R ``drtmle`` has no fold-wise reduced-regression construction.

**Reading, declared.**  ``publication_policy="gated"``.  Every primary row answers to the shared
``Margins()`` of ``performance-tests.csv``.  A red bias row is a defect suspect.
:data:`EXPECTED_FINITE_SAMPLE` names ``ey[1][V=2]`` before the run: its pooled coverage
(:data:`POOLED_SMALL_STRATUM_COVERAGE`) is below the floor, so no budget makes it pass, and the
budget is sized over the other rows.  If it is red, :func:`small_stratum_reading` reads it: an
SE-ratio interval containing 1, a coverage interval containing the pooled coverage and a bias
row that passes make it ``expected finite-sample``, owned by ``X8-drtmle-small-stratum``;
anything else is a defect suspect.  Any other red primary row is a defect suspect: the
declaration names no other finite-sample owner.

**Declared before any run** (the Fix-6 declaration).  The primary budget is 4,000
replications (:data:`PRIMARY_REPLICATES`), the budget :func:`rule` gives over the committed
pilot summaries (:data:`~tests.studies.drtmle_foldwise_sizing.PLANNING_PATH`) on
:data:`SIZED_ESTIMANDS`: predictive pass 0.957, target 0.95.  ``ey[1][V=2]`` is excluded from
the sizing and read as above.  No other row is set aside.  The ladder's budgets, its reading and
the ``X8-drtmle-one-sided-bias`` acceptance are declared in
:mod:`tests.studies.stratified_drtmle_foldwise_ladder`.  The primary phase costs about
:data:`CPU_HOURS` CPU-hours at the pilot's seconds per fit.

The diagnosis item of the plan review is closed: the ``both_correct`` marginal ATE, 0.19 SD in
the 200-draw pilot, reads 0.035 SD on the pilot's 1,000 draws, below the 0.12 SD at which it
would have been diagnosed before this declaration.

Every coverage row, the ladder's rungs included, gates the interval the library ships at this
declaration, the normal Wald interval on the identity scale
(:func:`tests.studies.evidence.schema.reported_inference`); a later change to the shipped
interval is re-read by that row's owner.  A property row carries ``estimate``, ``std_error``,
``truth`` and ``covered`` and no bounds: its identity-scale bounds are ``estimate`` plus or
minus ``z * std_error`` from those committed columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_stratified_drtmle as canonical
from tests.studies import drtmle_foldwise as foldwise
from tests.studies.drtmle_foldwise_sizing import (
    LadderPlan,
    Pilot,
    Rule,
    committed_planning,
    planned,
    without,
)
from tests.studies.evidence import replication
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS as PRIMARY_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import primary_rows

IMPLEMENTATION = foldwise.IMPLEMENTATION
#: ``canonical-stratified-drtmle``'s seed, on purpose: replication ``r`` of the scenario is that
#: study's registered replication ``r``.
SEED = canonical.SEED
#: Fresh entropy for every bootstrap of this record.
RESAMPLING_SEED = 2026111213
PRIMARY_N = canonical.PRIMARY_N
SCENARIO = canonical.SCENARIO
#: The marginal ATE's package name, the row the canonical study does not pair.
MARGINAL = canonical.ate_name("marginal_ate")
ESTIMANDS: tuple[str, ...] = (*canonical.ESTIMANDS, MARGINAL)
#: The replications ``canonical-stratified-drtmle`` registered, which this record pairs with.
REGISTERED_REPLICATES = canonical.PRIMARY_REPLICATES
FOLDWISE = foldwise.FOLDWISE

#: The primary row declared before the run as an expected finite-sample row (module docstring).
EXPECTED_FINITE_SAMPLE = ("ey[1][V=2]",)
#: The rows the primary budget is sized over.
SIZED_ESTIMANDS = tuple(name for name in ESTIMANDS if name not in EXPECTED_FINITE_SAMPLE)
#: ``ey[1][V=2]``'s coverage in ``canonical-stratified-drtmle``'s ``performance-tests.csv``
#: (pooled, n = 2,000, R = 800), the reading a fold-wise red row must be consistent with.
POOLED_SMALL_STRATUM_COVERAGE = 0.9
SMALL_STRATUM_CLEAN = "clean"
SMALL_STRATUM_FINITE_SAMPLE = "expected finite-sample"
SMALL_STRATUM_DEFECT = "defect suspect"


def small_stratum_reading(row: Mapping[str, Any]) -> str:
    """The declared reading of an :data:`EXPECTED_FINITE_SAMPLE` row of ``performance-tests.csv``.

    Parameters
    ----------
    row : Mapping
        The row: ``passed``, ``bias_equivalent``, ``coverage_ci_lower``, ``coverage_ci_upper``,
        ``se_ratio_ci_lower`` and ``se_ratio_ci_upper``.

    Returns
    -------
    str
        :data:`SMALL_STRATUM_CLEAN` when the row passes; else
        :data:`SMALL_STRATUM_FINITE_SAMPLE` when its bias passes, its SE-ratio interval
        contains 1 and its coverage interval contains
        :data:`POOLED_SMALL_STRATUM_COVERAGE`; else :data:`SMALL_STRATUM_DEFECT`.
    """
    if bool(row["passed"]):
        return SMALL_STRATUM_CLEAN
    se_ratio = Interval(float(row["se_ratio_ci_lower"]), float(row["se_ratio_ci_upper"]))
    coverage = Interval(float(row["coverage_ci_lower"]), float(row["coverage_ci_upper"]))
    if (
        bool(row["bias_equivalent"])
        and se_ratio.contains(1.0)
        and coverage.contains(POOLED_SMALL_STRATUM_COVERAGE)
    ):
        return SMALL_STRATUM_FINITE_SAMPLE
    return SMALL_STRATUM_DEFECT


# ---------------------------------------------------------------------- the ladder's cells
FAMILY = "stratified_contraction"
SIZES = (2_000, 4_000, 8_000)
POSITIVE = "treatment_correct"
CONTROL = "both_wrong"
REGIMES = (POSITIVE, CONTROL)
LABELS = canonical.ATE_LABELS
#: The labels the slope reading always reads; the X8 acceptance names both.
READ_LABELS = ("marginal_ate", "v1_ate")


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

# --------------------------------------------------------------------------------- the sizes
#: The floor of the primary budget.
PRIMARY_FLOOR = 1_000
#: Set by the declaration commit from :func:`rule` and the ladder's rules over the committed
#: pilot summaries.
BUDGETS_DECLARED = True
#: The declared primary budget.  :func:`rule` gives it with predictive pass 0.957 over
#: :data:`SIZED_ESTIMANDS`.
PRIMARY_REPLICATES = 4_000
#: The seed of the rule's planning draws.
PLANNING_SEED = 2026111223
#: The pilot group whose summary plans the primary unit.
PLANNING_GROUP = "s_primary"
#: The primary phase's cost in CPU-hours at the pilot's seconds per fit
#: (:data:`~tests.studies.drtmle_foldwise_sizing.PLANNING_PATH`); the ladder's is
#: :data:`tests.studies.stratified_drtmle_foldwise_ladder.CPU_HOURS`.
CPU_HOURS = 8.4


def rule(planning: Mapping[str, Pilot | LadderPlan] | None = None) -> Rule:
    """The arm-pooled predictive rule of the primary unit, every sized estimand jointly.

    The pilot's :data:`EXPECTED_FINITE_SAMPLE` rows are dropped before the rule reads it.

    Parameters
    ----------
    planning : Mapping or None
        Pilot group to its summary (the rule reads :data:`PLANNING_GROUP`); the committed
        summaries when ``None``.

    Returns
    -------
    Rule
        The rule, with the record's family-wise target 0.95 on its single unit.
    """
    pilot = planned(planning, PLANNING_GROUP, Pilot)
    sized = without(pilot, [row for row in EXPECTED_FINITE_SAMPLE if row in pilot.rows])
    return Rule(
        pilots={"primary": sized},
        units={"primary": (("primary",), PRIMARY_FLOOR)},
        seed=PLANNING_SEED,
    )


def cpu_hours(seconds: Mapping[str, float] | None = None) -> float:
    """The primary phase's cost in CPU-hours at the pilot's mean seconds per fit.

    Parameters
    ----------
    seconds : Mapping or None
        Pilot group to its seconds per fit; the committed values when ``None``.

    Returns
    -------
    float
        The CPU-hours of :data:`PRIMARY_REPLICATES`.
    """
    per_fit = committed_planning()[1] if seconds is None else seconds
    return PRIMARY_REPLICATES * per_fit[PLANNING_GROUP] / 3_600


def assert_declared() -> None:
    """Refuse a declared run while the budgets are planning values.

    Raises
    ------
    RuntimeError
        While :data:`BUDGETS_DECLARED` is ``False``.
    """
    if not BUDGETS_DECLARED:
        raise RuntimeError(
            "stratified-drtmle-foldwise's budgets are planning values; the sizing pilot and "
            "the declaration commit set them before any declared run"
        )


# ------------------------------------------------------------------------- the registration
STUDY = StudyRecord(
    name="DR-TMLE with baseline strata and fold-wise reduced regressions",
    slug="stratified-drtmle-foldwise",
    artifacts=ROOT / "tests" / "canonical" / "stratified_drtmle_foldwise",
    document="docs/technical-reference/method-evidence/stratified-dr-tmle-foldwise.md",
    anchor="dr-tmle-with-baseline-strata-and-fold-wise-reduced-regressions",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: 0.01},
    modules=(
        "tests/studies/stratified_drtmle_foldwise.py",
        "tests/studies/drtmle_foldwise_sizing.py",
        "tests/studies/canonical_stratified_drtmle.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/discrete_law.py",
        "tests/studies/drtmle_foldwise.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/canonical_multi_arm_drtmle.py",
        "tests/studies/drtmle_foldwise_properties.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/multi_arm_drtmle_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/stratified_drtmle_foldwise_ladder.py",
    ),
    runner_module="tests.studies.stratified_drtmle_foldwise",
    properties_module="tests.studies.stratified_drtmle_foldwise_ladder",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION: dict[str, Any] = {
    **{
        key: value
        for key, value in canonical.CONFIGURATION.items()
        if key not in {"excluded_comparisons", "declared_difference", "folds"}
    },
    "reduced_crossfit": FOLDWISE,
    "folds": (
        "canonical-stratified-drtmle's unstratified ten-fold vector, drawn by "
        "cleverly.random_partition from the sample's seed alone"
    ),
    "draws": (
        "canonical-stratified-drtmle's seed and scenario, so replication r < 800 is that "
        "study's registered replication r"
    ),
    "added_rows": "the marginal ATE, against the law's exact mixture truth",
    "expected_finite_sample": list(EXPECTED_FINITE_SAMPLE),
    "ladder": (
        "treatment_correct and both_wrong at n = 2,000, 4,000 and 8,000; the treatment_correct "
        "n = 2,000 rung draws canonical-stratified-drtmle's X8 stream (property_sample, "
        "treatment_correct), so replication r < 800 is the X8 cells' replication r; every "
        "other rung draws a fresh named stream of this record"
    ),
    "comparator": "none: Python evidence, no comparator",
    "comparator_reason": "R drtmle has no fold-wise reduced-regression construction",
}


def fit_cleverly(frame: pd.DataFrame, configuration: str = "both_correct") -> Any:
    """``canonical-stratified-drtmle``'s registered fit with fold-wise reduced regressions.

    Parameters
    ----------
    frame : pandas.DataFrame
        One sample with its ``fold`` column.
    configuration : str
        The nuisance configuration; the primary phase fits ``"both_correct"``.

    Returns
    -------
    Any
        The single fitted result.
    """
    return canonical.fit_cleverly(frame, configuration, reduced_crossfit=FOLDWISE)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """``canonical-stratified-drtmle``'s own draw, with its supplied fold vector."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return canonical.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this record's declared seed."""
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
        result=fit_cleverly(frame),
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
        [(SCENARIO, r, n) for r in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(PRIMARY_COLUMNS)])
