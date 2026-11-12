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

**No comparator and no property cell.**  R ``drtmle`` has no fold-wise construction.  The
one-correct evidence the support row cites is the registered ``drtmle-foldwise`` cell
``double_robustness/multi_arm_drtmle__treatment_correct``, so no new one-correct cell runs.
The multi-arm law is not H1, and no theory claim is made for it: the record is read on its rows
alone.

**Reading, declared.**  ``publication_policy="gated"``.  Every row answers to the shared
``Margins()`` of ``performance-tests.csv``.  A red bias row is a defect suspect; a red
calibration row is read under the finite-sample rule (the 99% SE-ratio interval contains 1 and
the coverage is consistent with 0.95), and only then can it become a ``reporting`` row with an
owner named in the declaration.

**Budgets are planning values.**  :data:`PLANNING_REPLICATES` is the primaries' floor, and
:func:`assert_declared` refuses a declared run until the declaration commit sets the budget
from the arm-pooled predictive rule (:func:`rule`) over the sizing pilot's rows.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_multi_arm_drtmle as canonical
from tests.studies import drtmle_foldwise as foldwise
from tests.studies import multi_arm_common
from tests.studies.drtmle_foldwise_sizing import Pilot, Rule, pilot_from_rows
from tests.studies.evidence import property_verdicts, replication
from tests.studies.evidence.properties import REPLICATE_COLUMNS
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

# --------------------------------------------------------------------------------- the sizes
#: A planning budget, not a declared one: the primaries' floor of 1,000 (plan section 5 plans
#: 1,200 to 2,400).
PLANNING_REPLICATES = 1_000
#: ``False`` until the declaration commit sets the budget from :func:`rule` and the pilot.
BUDGETS_DECLARED = False
PRIMARY_REPLICATES = PLANNING_REPLICATES
#: The floor of the primary budget.
PRIMARY_FLOOR = 1_000
#: The seed of the rule's planning draws.
PLANNING_SEED = 2026111222


@cache
def registered_pilot() -> Pilot:
    """The planning summary of the registered pooled primary rows.

    ``canonical-multi-arm-drtmle``'s subject rows at n = 2,000, R = 800, fitted with
    ``reduced_crossfit="pooled"``.  The X83 probe P1 read fold-wise ``|bias|`` at or below
    pooled ``|bias|`` in every regime, so a pooled row is an upper planning value for its
    fold-wise row.  The sizing pilot's fold-wise rows replace it.

    Returns
    -------
    Pilot
        One row per estimand, with its SE ratio and coverage.
    """
    rows = pd.read_parquet(canonical.STUDY.artifact("replicates.parquet"))
    own = rows.loc[rows["implementation"] == canonical.STUDY.implementation]
    return pilot_from_rows(
        own,
        source="canonical-multi-arm-drtmle replicates.parquet, pooled, n = 2,000",
        estimands=ESTIMANDS,
    )


def rule(pilot: Pilot | None = None) -> Rule:
    """The arm-pooled predictive rule of this record's one unit, every estimand jointly.

    Parameters
    ----------
    pilot : Pilot or None
        The planning summary; the registered pooled rows when ``None``.

    Returns
    -------
    Rule
        The rule, with the record's family-wise target 0.95 on its single unit.
    """
    chosen = registered_pilot() if pilot is None else pilot
    return Rule(
        pilots={"primary": chosen},
        units={"primary": (("primary",), PRIMARY_FLOOR)},
        seed=PLANNING_SEED,
    )


def assert_declared() -> None:
    """Refuse a declared run while the budget is a planning value.

    Raises
    ------
    RuntimeError
        While :data:`BUDGETS_DECLARED` is ``False``.
    """
    if not BUDGETS_DECLARED:
        raise RuntimeError(
            "multi-arm-drtmle-foldwise's budget is a planning value; the sizing pilot and the "
            "declaration commit set it before any declared run"
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
        "study's registered replication r"
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


# ------------------------------------------------------------------- no property cell
#: The share of a cell's requested replications the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """No property cell, so no row."""
    del n_jobs
    return pd.DataFrame(columns=list(REPLICATE_COLUMNS))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """No property cell, so no verdict."""
    if not rows.empty:
        raise ValueError(f"{STUDY.slug} declares no property cell, and it received rows")
    return pd.DataFrame(
        {
            "property": pd.Series(dtype=str),
            "cell": pd.Series(dtype=str),
            "role": pd.Series(dtype=str),
            "passed": pd.Series(dtype=bool),
            "property_passed": pd.Series(dtype=bool),
        }
    )
