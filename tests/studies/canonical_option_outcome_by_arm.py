r"""Declared evidence for ``outcome_by_arm=True`` on point TMLE (``option-outcome-by-arm``).

The subject is ``TMLE(outcome_by_arm=True)``: one outcome regression per treatment arm, fitted
on the arm's rows and the covariates alone (X76 part (b), Fix-8 plan sections 2.2 and 5.3).
The option changes the outcome model class and nothing else, so the remainder is the ordinary
product and double robustness is unchanged.  The study measures the interval of the per-arm
fit where a pooled main-terms regression is misspecified.  Laws, learners and limits are
:mod:`tests.studies._option_by_arm_law`; the property families are
:mod:`tests.studies.option_outcome_by_arm_properties`:

=============================  ==============================================================
family                         cells
=============================  ==============================================================
``interval_calibration`` (K)   per-arm GLMs, ATE: in sample on ``LB`` (binary) and ``LB-c``
                               (continuous); stacked over 5 folds on ``LB`` and ``LB-b``
                               (bounded, declared ``q_bounds``): 4 gated positives, each with
                               the derived ``shrunken_se_control`` and ``noise_control``
``double_robustness``          ``LB``: per-arm intercepts with the correct mechanism
                               (positive); per-arm intercepts and an intercept-only
                               mechanism (control)
``pooled_outcome`` (reported)  the pooled main-terms fit (``outcome_by_arm=False``) of each K
                               cell, on the K cell's draws
=============================  ==============================================================

**No comparator.**  R ``drtmle(stratify=TRUE)`` fits the same per-arm regressions; the
``drtmle_stratify.json`` fixture checks that path.  The study reads every cell against its exact
truth.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/option_outcome_by_arm/regenerate.py`` before any run.

**Registered.**  The declared run is committed at c957150e02, and the record is in
``registered()``.  The ``T53`` support rows and the evidence page landed with the registration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _option_by_arm_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import initial_estimates, primary_rows

SLUG = "option-outcome-by-arm"
SEED = 20267531
RESAMPLING_SEED = 20267532
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the in-sample binary K fit on ``LB`` at n = 2,000.
PRIMARY_SCENARIO = "LB"
PRIMARY_SETTING = "binary_in_sample"
PRIMARY_N = 2_000
#: Plan 5.3's K budget, above the 1,000 a primary needs.
PRIMARY_REPLICATES = 2_800
PRIMARY_ESTIMANDS = ("ate", "ey1", "ey0")

#: The K labels; each is a calibration cell and a ``pooled_outcome`` cell.
CALIBRATION_LABELS = (
    "in_sample_binary",
    "stacked_binary",
    "in_sample_continuous",
    "stacked_bounded",
)

#: Filled by :mod:`tests.studies.option_outcome_by_arm_properties`, which this module cannot
#: import; the design test pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "double_robustness": ("both_wrong", "treatment_correct"),
    "pooled_outcome": CALIBRATION_LABELS,
}

STUDY = StudyRecord(
    name="per-arm outcome regressions on point TMLE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "option_outcome_by_arm",
    document="docs/technical-reference/method-evidence/option-outcome-by-arm.md",
    anchor="per-arm-outcome-regressions-on-point-tmle",
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
        "tests/studies/canonical_option_outcome_by_arm.py",
        "tests/studies/option_outcome_by_arm_properties.py",
        "tests/studies/_option_by_arm_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/repeated_sizing.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/fluctuation/iterative.py",
        "src/cleverly/inference/influence.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_option_outcome_by_arm",
    properties_module="tests.studies.option_outcome_by_arm_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "in-sample and stacked point TMLE with outcome_by_arm=True",
    "laws": "W1, W2 ~ N(0, 1) iid; g0 = expit(0.4 W1 + 0.4 W2); "
    "eta1 = -0.2 + 0.9 W1 + 0.2 W2, eta0 = -0.5 + 0.1 W1 + 0.7 W2",
    "law_LB": "Y | A=a, W ~ Bernoulli(expit(eta_a))",
    "law_LB-c": "Y | A=a, W ~ N(eta_a, 1)",
    "law_LB-b": "Y | A=a, W ~ Beta(24 m, 24 (1 - m)), m = expit(eta_a)",
    "outcome_learners": {
        "by_arm": "per arm, on W: unpenalized logistic GLM (LB), linear regression (LB-c), "
        "fractional logistic GLM (LB-b), by Newton or least squares",
        "by_arm_intercept": "per arm, intercept only",
        "pooled": "one main-terms regression on (A, W1, W2) of the same family",
    },
    "treatment_learners": {
        "glm": "unpenalized logistic GLM on (W1, W2)",
        "intercept": "intercept only",
    },
    "estimands": list(law.ESTIMANDS),
    "g_bounds": list(law.G_BOUNDS),
    "q_bounds": f"declared {list(law.Q_BOUNDS)} on LB-b only",
    "max_iter": law.MAX_ITER,
    "tol": law.TOL,
    "cross_fit": f"False, except the stacked cells ({law.N_FOLDS} folds)",
    "simultaneous_intervals": False,
    "settings": {name: vars(setting) for name, setting in law.SETTINGS.items()},
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
    """Convert one fit to the shared replication schema."""
    return primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=PRIMARY_ESTIMANDS,
        initials=initial_estimates(result, PRIMARY_ESTIMANDS),
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(law.fit(frame, PRIMARY_SETTING), truth, scenario, replicate)


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
    "CALIBRATION_LABELS",
    "PROPERTY_CELLS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
