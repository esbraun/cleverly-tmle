r"""Fix-6 unit M: the fold-wise DR-TMLE default on the ``canonical-multi-arm-drtmle`` primary.

The subject is ``canonical-multi-arm-drtmle``'s registered fit,
:func:`tests.studies.canonical_multi_arm_drtmle.fit_cleverly`, with one setting changed:
``reduced_crossfit="foldwise"``, the shipped default.  Everything else is that study's primary
configuration: the three-arm binary law, n = 2,000, its unpenalized logistic learners, five
supplied unstratified folds, the univariate reduction, guard ``("Q", "g")``, and every estimand
of ``ALL_ESTIMANDS`` (three arm means and the ATE, RR and OR contrasts against ``high``).  The
record is the truth, coverage and SE-ratio evidence a support row for the default cross-fitted
multi-arm fit needs (Fix-6 D6).

**Pairing.**  :data:`SEED` **is** ``canonical-multi-arm-drtmle``'s seed and the scenario is its
scenario, so replication ``r < 800`` is that study's registered replication ``r``: the same rows
and the same fold vector.  Replications 800 and above continue the stream.
:data:`RESAMPLING_SEED` is fresh.

**The property cell.**  ``double_robustness/outcome_correct`` refits
``canonical-multi-arm-drtmle``'s registered ``double_robustness/outcome_correct`` cell (its
law, nuisances, n = 1,000 and stream root, :func:`registered_cell`) with
``reduced_crossfit="foldwise"`` (:func:`cell_settings`).  The registered cell's sample seeds are
the head of this cell's stream, so replication ``r < 600`` is its replication ``r``.  With the
registered ``drtmle-foldwise`` cell ``double_robustness/multi_arm_drtmle__treatment_correct``,
both one-correct branches of the multi-arm default are measured.  The cell reads the shared
``double_robustness`` rule.

**No comparator.**  R ``drtmle`` has no fold-wise construction.  The multi-arm law is not H1,
and no theory claim is made for it: the record is read on its rows alone.

**Reading, declared.**  ``publication_policy="gated"``.  Every row answers to the shared
``Margins()`` of ``performance-tests.csv``.  A red bias row is a defect suspect; a red
calibration row is read under the finite-sample rule (the 99% SE-ratio interval contains 1 and
the coverage interval contains 0.95, with the bias passing), and only then does it read
``finite-sample`` and go to the owner the declaration names,
:data:`~tests.studies.drtmle_foldwise_sizing.FINITE_SAMPLE_OWNER` (``Fix6-finite-sample``).
Every other red row is a defect suspect and is diagnosed.

**Declared before any run** (the Fix-6 declaration).  The budgets (:data:`BUDGETS`) are
1,200 primary replications and 800 for the cell, the budgets :func:`rule` gives over the
committed pilot summaries (:data:`~tests.studies.drtmle_foldwise_sizing.PLANNING_PATH`):
predictive pass 0.990 for the primary rows and 0.997 for the cell, family-wise target 0.95.  No
row is set aside.  The run costs about :data:`CPU_HOURS` CPU-hours at the pilot's seconds per
fit.

Every coverage row gates the interval the library ships at this declaration, the normal Wald
interval (:func:`tests.studies.evidence.schema.reported_inference`): on the identity scale for
the arm means, the ATE contrasts and the cell, and on the log scale for the RR and OR
contrasts, as the primary rows' ``inference_scale`` records.  A later change to the shipped
interval is re-read by this row's owner.  The property row carries ``estimate``,
``std_error``, ``truth`` and ``covered`` and no bounds: its identity-scale bounds are
``estimate`` plus or minus ``z * std_error`` from those committed columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.estimators import DRTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_multi_arm_drtmle as canonical
from tests.studies import drtmle_foldwise as foldwise
from tests.studies import multi_arm_common
from tests.studies import multi_arm_drtmle_properties as registered_properties
from tests.studies.drtmle_foldwise_sizing import (
    LadderPlan,
    Pilot,
    Rule,
    committed_planning,
    planned,
)
from tests.studies.evidence import property_verdicts, replication
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyBatch, PropertyCell
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS as PRIMARY_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

IMPLEMENTATION = foldwise.IMPLEMENTATION
#: ``canonical-multi-arm-drtmle``'s seed, on purpose: replication ``r`` of the scenario is that
#: study's registered replication ``r``.
SEED = canonical.SEED
#: Fresh entropy for every bootstrap of this record.
RESAMPLING_SEED = 2026111212
PRIMARY_N = canonical.PRIMARY_N
SCENARIO = canonical.SCENARIO
ESTIMANDS = multi_arm_common.ALL_ESTIMANDS
#: The replications ``canonical-multi-arm-drtmle`` registered, which this record pairs with.
REGISTERED_REPLICATES = canonical.PRIMARY_REPLICATES
FOLDWISE = foldwise.FOLDWISE

#: The property family and the cell, named as the registered cell is.
FAMILY = "double_robustness"
CELL = "outcome_correct"

# --------------------------------------------------------------------------------- the sizes
#: The floors the sizing rule starts from: ``primary`` at the primaries' 1,000 and
#: ``outcome_correct`` at the registered cell's 600.
FLOORS: Mapping[str, int] = {"primary": 1_000, CELL: 600}
#: Set by the declaration commit from :func:`rule` over the committed pilot summaries.
BUDGETS_DECLARED = True
#: The declared budgets.  :func:`rule` gives them with predictive pass 0.990 (``primary``) and
#: 0.997 (``outcome_correct``).
BUDGETS: Mapping[str, int] = {"primary": 1_200, CELL: 800}
PRIMARY_REPLICATES = BUDGETS["primary"]
#: The seed of the rule's planning draws.
PLANNING_SEED = 2026111222
#: The pilot groups whose summaries plan each unit of :func:`rule`.
PLANNING_GROUPS: Mapping[str, str] = {"primary": "m_primary", CELL: "m_outcome_correct"}
#: The run's cost in CPU-hours at the pilot's seconds per fit
#: (:data:`~tests.studies.drtmle_foldwise_sizing.PLANNING_PATH`).
CPU_HOURS = 0.6


def rule(planning: Mapping[str, Pilot | LadderPlan] | None = None) -> Rule:
    """The predictive rule of the record's family: the primary rows and the cell, jointly.

    Two units share the family-wise target 0.95: ``primary`` (every estimand, from the
    primaries' floor) and ``outcome_correct`` (the cell's bias, from its registered 600).

    Parameters
    ----------
    planning : Mapping or None
        Pilot group to its summary (:data:`PLANNING_GROUPS` names the groups read); the
        committed summaries when ``None``.

    Returns
    -------
    Rule
        The rule.
    """
    return Rule(
        pilots={name: planned(planning, group, Pilot) for name, group in PLANNING_GROUPS.items()},
        units={name: ((name,), FLOORS[name]) for name in PLANNING_GROUPS},
        seed=PLANNING_SEED,
    )


def cpu_hours(seconds: Mapping[str, float] | None = None) -> float:
    """The declared run's cost in CPU-hours at the pilot's mean seconds per fit.

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
    return sum(BUDGETS[name] * per_fit[group] for name, group in PLANNING_GROUPS.items()) / 3_600


def assert_declared() -> None:
    """Refuse a declared run while the budgets are planning values.

    Raises
    ------
    RuntimeError
        While :data:`BUDGETS_DECLARED` is ``False``.
    """
    if not BUDGETS_DECLARED:
        raise RuntimeError(
            "multi-arm-drtmle-foldwise's budgets are planning values; the sizing pilot and "
            "the declaration commit set them before any declared run"
        )


# ------------------------------------------------------------------------- the registration
STUDY = StudyRecord(
    name="multi-arm DR-TMLE with fold-wise reduced regressions",
    slug="multi-arm-drtmle-foldwise",
    artifacts=ROOT / "tests" / "canonical" / "multi_arm_drtmle_foldwise",
    document="docs/technical-reference/method-evidence/multi-arm-dr-tmle-foldwise.md",
    anchor="multi-arm-dr-tmle-with-fold-wise-reduced-regressions",
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
        "tests/studies/multi_arm_drtmle_foldwise.py",
        "tests/studies/drtmle_foldwise_sizing.py",
        "tests/studies/canonical_multi_arm_drtmle.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/drtmle_foldwise.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/discrete_law.py",
        "tests/studies/canonical_stratified_drtmle.py",
        "tests/studies/drtmle_foldwise_properties.py",
        "tests/studies/multi_arm_drtmle_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
    ),
    runner_module="tests.studies.multi_arm_drtmle_foldwise",
    properties_module="tests.studies.multi_arm_drtmle_foldwise",
    property_cells={FAMILY: (CELL,)},
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION: dict[str, Any] = {
    **{key: value for key, value in canonical.CONFIGURATION.items() if key != "folds"},
    "reduced_crossfit": FOLDWISE,
    "folds": (
        "canonical-multi-arm-drtmle's unstratified five-fold vector, drawn by "
        "cleverly.random_partition from the sample's seed alone"
    ),
    "draws": (
        "canonical-multi-arm-drtmle's seed and scenario, so replication r < 800 is that "
        "study's registered replication r; the outcome_correct cell draws the registered "
        "cell's stream, so replication r < 600 is that cell's replication r"
    ),
    "comparator": "none: Python evidence, no comparator",
    "comparator_reason": "R drtmle has no fold-wise reduced-regression construction",
}


def fit_cleverly(frame: pd.DataFrame, scenario: str = SCENARIO) -> Any:
    """``canonical-multi-arm-drtmle``'s registered fit with fold-wise reduced regressions.

    Parameters
    ----------
    frame : pandas.DataFrame
        One sample with its ``fold`` column.
    scenario : str
        The scenario; the fit does not depend on it.

    Returns
    -------
    Any
        The single fitted result.
    """
    return canonical.fit_cleverly(frame, scenario, reduced_crossfit=FOLDWISE)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """``canonical-multi-arm-drtmle``'s own draw, with its supplied fold vector."""
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
    scenario, replicate, n = key
    frame, truth = drawn
    return multi_arm_common.rows_from_result(
        fit_cleverly(frame, scenario),
        truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        n=n,
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


# ------------------------------------------------------------------------- the property cell
def registered_cell() -> PropertyCell:
    """``canonical-multi-arm-drtmle``'s registered ``double_robustness/outcome_correct`` cell."""
    (cell,) = [
        cell
        for cell in registered_properties.cells()
        if (cell.property, cell.cell) == (FAMILY, CELL)
    ]
    return cell


def property_cell(budget: int | None = None) -> PropertyCell:
    """The cell: the registered cell's law, nuisances, size and root, at this record's budget.

    Parameters
    ----------
    budget : int or None
        A smoke run's cap on the replications; the declared budget when ``None``.

    Returns
    -------
    PropertyCell
        The cell.
    """
    registered = registered_cell()
    declared = BUDGETS[CELL]
    return PropertyCell(
        FAMILY,
        CELL,
        registered.dgp,
        registered.outcome_learner,
        registered.treatment_learner,
        registered.n,
        declared if budget is None else min(budget, declared),
        registered.seed,
        role="positive",
        estimand=registered.estimand,
    )


def cell_settings(cell: PropertyCell) -> dict[str, Any]:
    """The registered cell's estimator arguments, with the fold-wise reduced regressions."""
    return {**registered_properties._settings(cell), "reduced_crossfit": FOLDWISE}


def _cell_estimator(cell: PropertyCell) -> Any:
    return lambda: DRTMLE(**cell_settings(cell))


def declared_cells() -> tuple[PropertyCell, ...]:
    """The cells :func:`generate_property_rows` runs, named for the truth-binding check."""
    return (property_cell(),)


#: The share of a cell's requested replications the accounting lets a run lose, declared
#: before the record's first run.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every replication of the cell; ``budget`` caps it for a smoke run.

    A declared run (``budget=None``) passes :func:`assert_declared` first.
    """
    if budget is None:
        assert_declared()
    rows = PropertyBatch(CELL, (property_cell(budget),), _cell_estimator).run(n_jobs=n_jobs)
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared ``double_robustness`` rule."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    return finish(summary, rates)
