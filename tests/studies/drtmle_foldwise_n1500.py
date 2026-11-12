r"""Fix-6 unit B: the fold-wise DR-TMLE default on the RM18 binary cells' own draws, n = 1,500.

The subject is the shipped cross-fitted :class:`~cleverly.estimators.DRTMLE` with
``reduced_crossfit="foldwise"``, fitted by :func:`tests.studies.drtmle_foldwise.fit_cleverly`
at :data:`tests.studies.drtmle_foldwise.PRIMARY_FIT`: the ``canonical-drtmle`` learners, ten
folds, the univariate reduction, guard ``("Q", "g")`` and the fold-wise reduced regressions.
The two ``canonical-drtmle`` cells the ``RM18-one-sided-bias`` owner holds,
``double_robustness/outcome_correct`` and ``double_robustness/treatment_correct``, fit
``reduced_crossfit="pooled"`` at n = 1,500.  The fold-wise default was measured at n = 1,000
and 3,000 only (``drtmle-foldwise``), so this record fits it at the red cells' own size.

**Pairing.**  Replication ``r`` of each cell draws
``stream_seed(canonical-drtmle, "property", "double_robustness", <scenario>, str(r + offset))``
with the offset :func:`tests.studies.drtmle_properties.cells` declares (11,000 for
``outcome_correct``, 12,000 for ``treatment_correct``).  Replication ``r < 800`` is therefore
the RM18 cell's registered replication ``r``: the same rows and the same fold vector.  A reader
pairs the two ``property-replicates.parquet`` files on ``(cell, replicate)`` to read the
pooled-minus-fold-wise displacement at n = 1,500; it is published, not gated.  Replications 800
and above continue the same stream.  :data:`RESAMPLING_SEED` is fresh, so every bootstrap of
this record is drawn from a stream no other study uses.

**The primary phase.**  Both regimes at n = 1,500 on this record's own fresh seed
(:data:`SEED`): the truth, coverage and SE-ratio rows of ``ate``, ``ey0`` and ``ey1``, gated by
the shared ``Margins()`` of ``performance-tests.csv``.  Fresh draws and the RM18 cells' own draws
both measure the default, so a verdict selected on the red cells' draws cannot carry the RM18
closure alone: the acceptance reads "when the primary rows and both paired cells pass".

**Reading, declared.**  Each cell is ``role="positive"`` in the shared ``double_robustness``
family: its 99% interval of the standardized bias inside 0.25 empirical SD, with the reported
SE inside the union-model band.  The pooled-minus-fold-wise displacement is published, not
gated.  ``publication_policy="gated"``.  A red row is a defect suspect, diagnosed and fixed
under a fresh declaration; no replication is added after a verdict.

**Budgets are planning values.**  :data:`PLANNING_BUDGETS` holds them, and
:func:`assert_declared` refuses a declared run until the declaration commit sets
:data:`BUDGETS_DECLARED` from the sizing rule (:func:`rule`), which sizes the primary rows and
the paired cells as one family.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_drtmle as canonical
from tests.studies import drtmle_foldwise as foldwise
from tests.studies import drtmle_properties as rm18
from tests.studies.drtmle_foldwise_sizing import Pilot, Rule, pilot_from_rows, single
from tests.studies.evidence import property_verdicts, replication
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.schema import REPLICATE_COLUMNS as PRIMARY_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed
from tests.studies.point_study_helpers import initial_estimates, primary_rows

IMPLEMENTATION = foldwise.IMPLEMENTATION
#: The primary phase's own seed, fresh: no registered study draws from it.
SEED = 2026111201
#: Fresh entropy for every bootstrap of this record.
RESAMPLING_SEED = 2026111211
PRIMARY_N = 1_500
#: The RM18 cells this record refits, in ``canonical-drtmle``'s order.
SCENARIOS = ("outcome_correct", "treatment_correct")
ESTIMANDS = canonical.ESTIMANDS
PRIMARY_SCENARIOS = SCENARIOS
#: The family and the cells, named as the RM18 cells are.
FAMILY = "double_robustness"
#: The replications each RM18 cell registered, which replications ``r < 800`` pair with.
REGISTERED_REPLICATES = rm18.DOUBLE_ROBUST_REPLICATES

# --------------------------------------------------------------------------------- the sizes
#: Planning budgets, not declared ones: ``primary`` per regime at the primaries' floor, and
#: ``cells`` per cell at the paired floor (each RM18 cell's 800).  The sizing rule may raise
#: either.
PLANNING_BUDGETS: Mapping[str, int] = {"primary": 1_000, "cells": 800}
#: ``False`` until the declaration commit sets the budgets from :func:`rule` and the pilot.
BUDGETS_DECLARED = False
#: The budgets the run uses.  They equal :data:`PLANNING_BUDGETS` until the declaration.
BUDGETS: Mapping[str, int] = PLANNING_BUDGETS
PRIMARY_REPLICATES = BUDGETS["primary"]
#: The seed of the rule's planning draws.
PLANNING_SEED = 2026111221

#: The planning standardized bias of each regime's ``ate`` at n = 1,500, before the sizing
#: pilot: ``treatment_correct`` about 0.12 SD, between the fold-wise 0.156 at n = 1,000 and
#: 0.072 at 3,000 (``drtmle-foldwise``); ``outcome_correct`` about 0.08 SD, the pooled 0.19 at
#: n = 1,500 less a displacement of about 0.11 scaled from 0.076 at 3,000.  Each carries the
#: planning uncertainty of 800 replications, the RM18 cells' budget.  The pilot replaces both.
PLANNING_ATE_BIAS: Mapping[str, float] = {"outcome_correct": 0.08, "treatment_correct": 0.12}
PLANNING_SOURCE = "Fix-6 planning values, interpolated"


@cache
def planning_pilots() -> dict[str, Pilot]:
    """The planning summaries of the primary rows and the cells, before the sizing pilot.

    A primary pilot is ``drtmle-foldwise``'s fold-wise primary of the regime at n = 3,000
    (R = 6,000): its correlation, SE ratios and coverages as they are, ``ate`` at
    :data:`PLANNING_ATE_BIAS`, and ``ey0`` and ``ey1`` scaled by ``sqrt(2)`` to n = 1,500, as a
    second-order bias is.  A cell pilot is ``ate`` at :data:`PLANNING_ATE_BIAS`, read on its
    bias alone.  Every planning value carries the uncertainty of 800 replications.

    Returns
    -------
    dict
        ``primary_<regime>`` and ``cell_<regime>`` for each regime.
    """
    rows = pd.read_parquet(foldwise.STUDY.artifact("replicates.parquet"))
    out: dict[str, Pilot] = {}
    for scenario in SCENARIOS:
        measured = pilot_from_rows(
            rows.loc[rows["scenario"] == scenario],
            source=f"drtmle-foldwise replicates.parquet, {scenario}, n = 3,000",
            estimands=ESTIMANDS,
        )
        bias = tuple(
            PLANNING_ATE_BIAS[scenario] if row == "ate" else round(value * float(np.sqrt(2)), 4)
            for row, value in zip(measured.rows, measured.bias, strict=True)
        )
        out[f"primary_{scenario}"] = dataclasses.replace(
            measured, source=f"{PLANNING_SOURCE}; {measured.source}", replicates=800, bias=bias
        )
        out[f"cell_{scenario}"] = single(PLANNING_SOURCE, 800, "ate", PLANNING_ATE_BIAS[scenario])
    return out


def rule(pilots: Mapping[str, Pilot] | None = None) -> Rule:
    """The predictive rule of the record's family: the primary rows and the cells, jointly.

    Two units share the family-wise target 0.95: ``primary`` (both regimes' primary rows, from
    the primaries' floor) and ``cells`` (both paired cells, from the paired floor).

    Parameters
    ----------
    pilots : Mapping or None
        ``primary_<regime>`` and ``cell_<regime>`` summaries; :func:`planning_pilots` when
        ``None``.

    Returns
    -------
    Rule
        The rule.
    """
    chosen = planning_pilots() if pilots is None else dict(pilots)
    return Rule(
        pilots=chosen,
        units={
            "primary": (
                tuple(f"primary_{scenario}" for scenario in SCENARIOS),
                PLANNING_BUDGETS["primary"],
            ),
            "cells": (
                tuple(f"cell_{scenario}" for scenario in SCENARIOS),
                PLANNING_BUDGETS["cells"],
            ),
        },
        seed=PLANNING_SEED,
    )


def assert_declared() -> None:
    """Refuse a declared run while the budgets are planning values.

    Raises
    ------
    RuntimeError
        While :data:`BUDGETS_DECLARED` is ``False``.
    """
    if not BUDGETS_DECLARED:
        raise RuntimeError(
            "drtmle-foldwise-n1500 budgets are planning values; the sizing pilot and the "
            "declaration commit set them before any declared run"
        )


# ------------------------------------------------------------------------- the registration
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {FAMILY: SCENARIOS}

STUDY = StudyRecord(
    name="DR-TMLE fold-wise default on the RM18 binary draws",
    slug="drtmle-foldwise-n1500",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_foldwise_n1500",
    document="docs/technical-reference/method-evidence/dr-tmle-foldwise-n1500.md",
    anchor="dr-tmle-fold-wise-default-on-the-rm18-binary-draws",
    scenarios=dict.fromkeys(PRIMARY_SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: 0.01},
    modules=(
        "tests/studies/drtmle_foldwise_n1500.py",
        "tests/studies/drtmle_foldwise_sizing.py",
        "tests/studies/drtmle_foldwise.py",
        "tests/studies/drtmle_properties.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/discrete_law.py",
        "tests/studies/canonical_multi_arm_drtmle.py",
        "tests/studies/canonical_stratified_drtmle.py",
        "tests/studies/drtmle_foldwise_properties.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/multi_arm_drtmle_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
    ),
    runner_module="tests.studies.drtmle_foldwise_n1500",
    properties_module="tests.studies.drtmle_foldwise_n1500",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION: dict[str, Any] = {
    **{
        key: value
        for key, value in foldwise.CONFIGURATION.items()
        if key not in {"spline_qr_learners", "reduced_learner_limit", "draws"}
    },
    "draws": (
        "each cell draws the canonical-drtmle RM18 cell's property stream, so replication "
        "r < 800 is that cell's registered replication r; the primary phase draws this "
        "record's own fresh seed"
    ),
    "comparator_reason": "R drtmle has no fold-wise reduced-regression construction",
}


# --------------------------------------------------------------------- the primary phase
def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the paper law with ``canonical-drtmle``'s ten-fold vector."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return canonical.draw_from_seed(scenario, n, seed)


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
    result = foldwise.fit_cleverly(frame, scenario, foldwise.PRIMARY_FIT)
    return primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every primary replication; a declared budget passes the guard first."""
    if replicates >= PRIMARY_REPLICATES:
        assert_declared()
    outcomes = replication.fit_primary(
        [(scenario, r, n) for scenario in PRIMARY_SCENARIOS for r in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(PRIMARY_COLUMNS)])


# ---------------------------------------------------------------------- the paired cells
def registered_cell(scenario: str) -> rm18.Cell:
    """The RM18 cell of ``canonical-drtmle`` whose stream ``scenario``'s cell reads."""
    (cell,) = [cell for cell in rm18.cells() if (cell.property, cell.cell) == (FAMILY, scenario)]
    return cell


def sample_seed(scenario: str, replicate: int) -> int:
    """The seed of replication ``replicate`` of a cell: the RM18 cell's stream, continued."""
    cell = registered_cell(scenario)
    return stream_seed(
        canonical.STUDY, "property", cell.property, cell.cell, str(replicate + cell.seed_offset)
    )


_Key = tuple[tuple[str, int, int]]
_Drawn = tuple[pd.DataFrame, dict[str, float]]


def _draw_cell(key: _Key) -> _Drawn:
    """The sample a key fits, a function of the key alone."""
    scenario, replicate, _ = key[0]
    return canonical.draw_from_seed(scenario, PRIMARY_N, sample_seed(scenario, replicate))


def _cells_of(key: _Key) -> list[str]:
    return [f"{FAMILY}/{key[0][0]}"]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[0][1]


def _seed_of(key: _Key) -> int:
    return sample_seed(key[0][0], key[0][1])


def _fit_key(key: _Key, drawn: _Drawn) -> list[dict[str, Any]]:
    scenario, replicate, requested = key[0]
    frame, truth = drawn
    result = foldwise.fit_cleverly(frame, scenario, foldwise.PRIMARY_FIT)
    return [
        replicate_row(
            property_name=FAMILY,
            cell=scenario,
            role="positive",
            replicate=replicate,
            n=PRIMARY_N,
            requested=requested,
            truth=float(truth["ate"]),
            estimate=result.estimates["ate"],
            alpha=STUDY.margins.alpha,
        )
    ]


def payloads(budget: int | None = None) -> list[_Key]:
    """Every paired replication; ``budget`` caps each cell for a smoke run."""
    declared = BUDGETS["cells"]
    count = declared if budget is None else min(budget, declared)
    return [((scenario, r, count),) for scenario in SCENARIOS for r in range(count)]


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: before the record's first run.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every paired replication; ``budget`` caps each cell for a smoke run.

    A declared run (``budget=None``) passes :func:`assert_declared` first.
    """
    if budget is None:
        assert_declared()
    outcomes = fit_properties(
        payloads(budget),
        _fit_key,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw_cell,
        seed_of=_seed_of,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared ``double_robustness`` rule."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    return finish(summary, rates)
