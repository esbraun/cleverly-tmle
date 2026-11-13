r"""Declared evidence for ``patience=`` on the cross-validated C-TMLE path (``option-patience``).

The subject is ``CTMLE(patience=p)``: the cross-validated scoring of the candidate path stops
after ``p`` candidates whose risk is not a new minimum, and the selection is the smallest risk
among the scored candidates (X76 part (c), Fix-8 plan sections 2.3 and 5.5).  Selector paths
withhold the interval (F18), so the study reads point estimates, stop shares and paired
displacements.  The laws, learners and fits are :mod:`tests.studies._option_patience_laws`; the
property families are :mod:`tests.studies.option_patience_properties`:

============================  ===========================================  ===================
family                        cell                                         role
============================  ===========================================  ===================
``patience_bias``             ``LC1`` greedy, patience ``p*``              positive: bias
                                                                           inside 0.25 SD
``patience_witness``          the same fits                                positive: stop share
                                                                           lower limit >= 0.3
``ordered_late_confounder``   ``LC3`` ordered (instruments first),         control: bias beyond
                              patience 2                                   0.25 SD
``late_confounder_stop``      the same fits                                positive: share
                                                                           excluding ``W1``,
                                                                           lower limit >= 0.3
``ordered_whole_path``        ``LC3`` ordered, whole path, same draws      positive: bias
                                                                           inside 0.25 SD
``greedy_late_confounder``    ``LC3`` greedy, patience 2, minus the        positive: mean
                              greedy whole path on the same draw           paired difference
                                                                           inside 0.25 SD of
                                                                           the whole path
``reported_*``                every other fit: LC1 ordered and discrete,   diagnostic
                              LC2 lasso and learners, with patience and
                              the whole path; stop shares and paired
                              displacements
============================  ===========================================  ===================

**No comparator.**  R ``ctmle``'s ``ctmleDiscrete(patience=)`` counter is the implementation
reference (``tests/unit/test_patience.py`` transcribes it); the study reads every cell against
its exact truth.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/option_patience/regenerate.py`` before any run.

**Registered** from its committed run, with the ``C12.greedy`` support row, the limit on
``C01.greedy`` and the evidence page ``option-patience.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _option_patience_laws as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import primary_rows

SLUG = "option-patience"
SEED = 20267541
RESAMPLING_SEED = 20267542
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the gated greedy fit with patience ``p*`` on ``LC1`` at n = 2,000.
PRIMARY_SCENARIO = "lc1"
PRIMARY_FIT = "lc1_greedy_patience"
PRIMARY_N = law.N
#: The bias cells' budget (plan 5.5), above the 1,000 a primary needs.
PRIMARY_REPLICATES = 1_600
PRIMARY_ESTIMANDS = ("ate",)
#: Every property cell's replications (plan 5.5).
REPLICATES = 1_600

#: The witness shares' floor (plan 5.5): a gated share's exact lower limit is at least this.
WITNESS_SHARE = 0.3

#: Filled by :mod:`tests.studies.option_patience_properties`, which this module cannot import;
#: the design test pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "patience_bias": ("lc1_greedy",),
    "patience_witness": ("lc1_greedy",),
    "ordered_late_confounder": ("lc3_ordered",),
    "late_confounder_stop": ("lc3_ordered",),
    "ordered_whole_path": ("lc3_ordered",),
    "greedy_late_confounder": ("lc3_greedy",),
    "reported_estimate": (
        "lc1_discrete_patience",
        "lc1_discrete_whole",
        "lc1_greedy_whole",
        "lc1_ordered_patience",
        "lc1_ordered_whole",
        "lc2_lasso_patience",
        "lc2_lasso_whole",
        "lc2_learners_patience",
        "lc2_learners_whole",
        "lc3_greedy_patience",
        "lc3_greedy_whole",
    ),
    "reported_stop": ("lc1_discrete", "lc1_ordered", "lc2_lasso", "lc2_learners", "lc3_greedy"),
    "reported_displacement": (
        "lc1_discrete",
        "lc1_greedy",
        "lc1_ordered",
        "lc2_lasso",
        "lc2_learners",
    ),
}

STUDY = StudyRecord(
    name="Patience on the cross-validated C-TMLE path",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "option_patience",
    document="docs/technical-reference/method-evidence/option-patience.md",
    anchor="patience-on-the-cross-validated-c-tmle-path",
    scenarios={PRIMARY_SCENARIO: PRIMARY_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/studies/canonical_option_patience.py",
        "tests/studies/option_patience_properties.py",
        "tests/studies/_option_patience_laws.py",
        "tests/studies/canonical_ctmle_selector.py",
        "tests/studies/canonical_ctmle_sequences.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/point_study_helpers.py",
        "src/cleverly/estimators/ctmle.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/fluctuation/iterative.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_option_patience",
    properties_module="tests.studies.option_patience_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
    # The claim is points only (F18): the primary's interval columns are the plug-in
    # diagnostic of a selector path, published and not gated.
    point_only_scenarios=frozenset({PRIMARY_SCENARIO}),
)

CONFIGURATION = {
    "construction": "in-sample selector C-TMLE with patience= and with the whole path",
    "law_LC1": "the canonical-ctmle-selector binary law (W1, W2, W3), correct logistic GLMs",
    "law_LC2": "the canonical-ctmle-sequences lasso and learners laws (W1 to W4)",
    "law_LC3": "W1, Z1, Z2, Z3 ~ N(0, 1) iid; A ~ Bernoulli(expit(0.8 (Z1 + Z2 + Z3) + 0.6 W1)); "
    "Y ~ Bernoulli(expit(-0.3 + 0.5 A + W1))",
    "learners": {
        "LC1, LC2": "unpenalized logistic GLMs on every covariate (outcome) and on each "
        "candidate set (treatment); LC2 the sibling study's lasso grid and declared "
        "candidates, on a selection split drawn with each sample",
        "LC3": "outcome: unpenalized logistic on (A, Z1, Z2, Z3), omitting W1; treatment: "
        "unpenalized logistic on each candidate set",
    },
    "patience": {
        "LC1": law.LC1_PATIENCE,
        "LC2": law.LC2_PATIENCE,
        "LC3": law.LC3_PATIENCE,
    },
    "lc3_ordering": list(law.LC3_ORDERING),
    "g_bounds": {
        "LC1": list(law.LC1_G_BOUNDS),
        "LC2": list(law.LC2_G_BOUNDS),
        "LC3": list(law.LC3_G_BOUNDS),
    },
    "selection_folds": law.SELECTION_FOLDS,
    "selection_inner_folds": law.SELECTION_INNER_FOLDS,
    "penalty": False,
    "max_iter": law.MAX_ITER,
    "tol": law.TOL,
    "cross_fit": False,
    "simultaneous_intervals": False,
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(scenario, n, seed), law.truth(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema.

    The interval columns hold the plug-in diagnostic of the working mechanism
    (:func:`~tests.studies.evidence.schema.reported_inference`): a selector path withholds
    the interval (F18), so they make no coverage claim.
    """
    return primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=PRIMARY_ESTIMANDS,
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(law.fit(frame, PRIMARY_FIT), truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(PRIMARY_SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The exact truth of every primary estimand, by scenario."""
    truth = law.truth(PRIMARY_SCENARIO)
    return {PRIMARY_SCENARIO: {name: truth[name] for name in PRIMARY_ESTIMANDS}}


__all__ = [
    "PROPERTY_CELLS",
    "REPLICATES",
    "STUDY",
    "WITNESS_SHARE",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
