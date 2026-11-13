r"""Declared evidence for ``markov_order=`` on LTMLE (``option-markov-order``).

The subject is ``LTMLE`` fitted on ``LongitudinalData.with_markov_order(k)``: every nuisance at
node ``t`` reads ``W`` and a window of the last nodes (X76 part (f), Fix-8 plan sections 2.6 and
5.7).  The order restricts the observed-data model and does not change the estimand.  Under the
restriction the windowed nuisances can be correct and the interval is the ordinary LTMLE's; the
row claims validity, not efficiency.  Laws, learners and limits are
:mod:`tests.studies._option_markov_law`; the property families are
:mod:`tests.studies.option_markov_order_properties`:

==============================  ===========================================================
family                          cells
==============================  ===========================================================
``interval_calibration`` (K)    ``LF1`` (Markov order 1), order 1 on ``EY(always)`` and the
                                ATE (always vs never), order 2 on the ATE: 3 gated
                                positives, each with the derived controls
``order_necessity``             ``LF1`` at order 0 (both sides violated), ATE: gated control
``order_double_robustness``     ``LF2`` at order 0 (the outcome window wrong, the mechanisms
                                right), ATE: gated positive
``markov_reported``             the full-history fit on the K draws (``EY(always)``, ATE),
                                and ``EY(always)`` of each order-0 fit
==============================  ===========================================================

**No comparator.**  R ``ltmle`` has no window option.  The study reads every cell against its
exact truth.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/option_markov_order/regenerate.py`` before any run.

**Declared, not registered.**  The record is not in ``registered()`` until its run is
committed; the ``G57`` support row and the evidence page land with that registration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _option_markov_law as law
from tests.studies.canonical_ltmle import regimen_initials, regimen_rows
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "option-markov-order"
SEED = 20267651
RESAMPLING_SEED = 20267652
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the order-1 fit on ``LF1`` at n = 2,000.
PRIMARY_SCENARIO = "LF1"
PRIMARY_SETTING = "lf1_order1"
PRIMARY_N = 2_000
#: Plan 5.7's K budget, above the 1,000 a primary needs.
PRIMARY_REPLICATES = 2_800
PRIMARY_ESTIMANDS = law.ESTIMANDS

#: The K labels: the order, then the estimand.
CALIBRATION_LABELS = ("order1_ey", "order1_ate", "order2_ate")

#: Filled by :mod:`tests.studies.option_markov_order_properties`, which this module cannot
#: import; the design test pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "order_necessity": ("ate",),
    "order_double_robustness": ("ate",),
    "markov_reported": ("full_history_ey", "full_history_ate", "order0_ey", "lf2_order0_ey"),
}

STUDY = StudyRecord(
    name="Markov-order windows on LTMLE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "option_markov_order",
    document="docs/technical-reference/method-evidence/option-markov-order.md",
    anchor="option-markov-order",
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
        "tests/studies/canonical_option_markov_order.py",
        "tests/studies/option_markov_order_properties.py",
        "tests/studies/_option_markov_law.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/repeated_sizing.py",
        "src/cleverly/longitudinal/data.py",
        "src/cleverly/longitudinal/estimator.py",
        "src/cleverly/inference/influence.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_option_markov_order",
    properties_module="tests.studies.option_markov_order_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "in-sample LTMLE on LongitudinalData.with_markov_order(k)",
    "laws": "W ~ U(0, 1); four nodes of binary L_t, A_t, retention C_t; "
    "L_t ~ Bernoulli(0.15 + 0.3 W + 0.3 L_{t-1} - 0.1 A_{t-1}); "
    "C_t ~ Bernoulli(expit(2.8 + 0.3 W - 0.4 L_t)); "
    "Y ~ Bernoulli(0.15 + 0.15 W + 0.15 L_4 + 0.35 L_3 - 0.05 A_4 - 0.05 A_3)",
    "law_LF1": "A_t ~ Bernoulli(expit(-1.0 + 0.4 W - 0.5 L_t + 2.0 A_{t-1} + 1.5 L_{t-1}))",
    "law_LF2": "A_t ~ Bernoulli(expit(-1.0 + 0.4 W - 0.5 L_t + 2.0 A_{t-1}))",
    "discarded_law": "rev 2 LF1: A_t ~ Bernoulli(expit(-0.6 + 0.4 W - 0.5 L_t + 0.3 A_{t-1} "
    "+ 1.5 L_{t-1})); windowed and full-history EY(always) both covered 0.919 at n = 2,000",
    "regimens": dict(law.REGIMENS),
    "outcome_learner": "LinearRegression (and pseudo_learner)",
    "mechanism_learners": "unpenalized logistic regression by Newton (treatment, censoring)",
    "orders": {name: setting.order for name, setting in law.SETTINGS.items()},
    "g_bounds": list(law.G_BOUNDS),
    "max_iter": law.MAX_ITER,
    "tol": law.TOL,
    "n_folds": 1,
    "simultaneous_intervals": False,
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(scenario, n, seed), law.truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Fit one primary sample and convert it to the shared replication schema."""
    result = law.fit(frame, PRIMARY_SETTING)
    return regimen_rows(
        STUDY,
        result,
        truth,
        regimen_initials(result, law.REGIMENS, (law.ATE,)),
        PRIMARY_ESTIMANDS,
        scenario,
        replicate,
        n=len(frame),
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, replicate)


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
    truth = law.truth()
    return {PRIMARY_SCENARIO: {name: truth[name] for name in PRIMARY_ESTIMANDS}}


__all__ = [
    "CALIBRATION_LABELS",
    "PROPERTY_CELLS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
