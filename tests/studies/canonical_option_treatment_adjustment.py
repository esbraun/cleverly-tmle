r"""Declared evidence for ``treatment_adjustment=`` on point TMLE (``option-treatment-adjustment``).

The subject is ``TMLE`` fitted with ``treatment_covariates=("W1", "W2")``, the low-level form of
``PointTreatment(treatment_adjustment=...)``: the treatment mechanism reads the subset
:math:`W_g = (W_1, W_2)` of the adjustment set :math:`W = (W_1, W_2, Z)`, and the outcome
regression reads all of :math:`W` (X76 part (d), Fix-8 plan sections 2.4 and 5.4).

The declaration restricts the model.  Under the restriction the efficient influence curve of
the ATE is unchanged and the remainder is the ordinary product, so the Wald interval holds at
the product rate.  When the mechanism also depends on a left-out covariate the remainder gains
the term :math:`P_0[(1 - g_0/g_\infty)(Q^* - Q_0)]`, linear in the outcome-regression error,
and the interval loses its level.  The study measures both sides.  Laws, learners and limits
are :mod:`tests.studies._option_adjustment_laws`; the property families are
:mod:`tests.studies.option_treatment_adjustment_properties`:

=================================  ==========================================================
family                             cells
=================================  ==========================================================
``interval_calibration`` (K)       ``LDA`` (the restriction holds), correct logistic GLMs,
                                   ATE, in sample and stacked: 2 gated positives, each with
                                   the derived ``shrunken_se_control`` and ``noise_control``
``double_robustness``              ``LDA``: intercept-only outcome regression with the
                                   correct mechanism on :math:`W_g` (positive); intercept-only
                                   outcome regression and mechanism (control)
``instrument_slow_outcome`` (LR)   ``LDB`` (:math:`Z` an instrument, the restriction false),
                                   the outcome GLM plus the constructed error
                                   :math:`k n^{-1/4} Z`, the mechanism on :math:`W_g`, at
                                   n = 16,000: gated control, bias beyond the margin
reported                           the same fit at n = 1,000 and 4,000 and the slope of its
                                   log absolute bias; the same outcome fit with the
                                   mechanism on the whole :math:`W` (no declaration) at each
                                   n; ``LDB`` with the unperturbed outcome GLM at n = 2,000
=================================  ==========================================================

The ``instrument_slow_outcome`` ladder is the cell on a law where the restriction fails,
with an estimated mechanism, so the evidence can see the term the restriction hides: its
standardized bias grows like :math:`n^{1/4}`.

**No comparator.**  R ``ctmle`` takes ``Wg=`` only on its collaborative path; the candidate
pool is checked by the ``ctmle_wg.json`` fixture.  The study reads every cell against its exact
truth.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/option_treatment_adjustment/regenerate.py`` before any run.

**Registered.**  The declared run is committed at 8d2826ecc4, and the record is in
``registered()``.  The ``T54`` support rows and the evidence page landed with the registration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _option_adjustment_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import initial_estimates, primary_rows

SLUG = "option-treatment-adjustment"
SEED = 20267641
RESAMPLING_SEED = 20267642
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the K in-sample fit on ``LDA`` at n = 2,000, the arm means and the ATE.
PRIMARY_SCENARIO = "LDA"
PRIMARY_SETTING = "lda_in_sample"
PRIMARY_N = 2_000
#: Plan 5.4's K budget, above the 1,000 a primary needs.
PRIMARY_REPLICATES = 2_400
PRIMARY_ESTIMANDS = ("ate", "ey1", "ey0")

#: Filled by :mod:`tests.studies.option_treatment_adjustment_properties`, which this module
#: cannot import; the design test pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in ("in_sample", "stacked")
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "double_robustness": ("both_wrong", "treatment_correct"),
    "instrument_slow_outcome": ("n16000",),
    "instrument_slow_outcome_ladder": ("n1000", "n4000"),
    "instrument_slow_outcome_rate": ("bias_slope",),
    "instrument_reference": ("n1000", "n4000", "n16000"),
    "instrument_parametric": ("n2000",),
}

STUDY = StudyRecord(
    name="treatment-mechanism covariates on point TMLE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "option_treatment_adjustment",
    document="docs/technical-reference/method-evidence/option-treatment-adjustment.md",
    anchor="treatment-mechanism-covariates-on-point-tmle",
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
        "tests/studies/canonical_option_treatment_adjustment.py",
        "tests/studies/option_treatment_adjustment_properties.py",
        "tests/studies/_option_adjustment_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/repeated_sizing.py",
        "src/cleverly/data/treatment_covariates.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/fluctuation/iterative.py",
        "src/cleverly/inference/influence.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_option_treatment_adjustment",
    properties_module="tests.studies.option_treatment_adjustment_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "in-sample and stacked point TMLE with treatment_covariates=(W1, W2)",
    "law_LDA": "W1, W2, Z ~ N(0, 1) iid; g0 = expit(0.5 W1 - 0.5 W2); "
    "logit Qbar0 = -0.3 + 0.6 A + 0.7 W1 + 0.4 W2 + 0.8 Z",
    "law_LDB": "W1, W2, Z ~ N(0, 1) iid; g0 = expit(0.5 W1 - 0.5 W2 + Z); "
    "logit Qbar0 = -0.3 + 0.6 A + 0.7 W1 + 0.4 W2",
    "adjustment": list(laws.COVARIATES),
    "treatment_covariates": list(laws.SUBSET),
    "outcome_learners": {
        "glm": "unpenalized logistic GLM on (A, W1, W2, Z), by Newton",
        "intercept": "intercept only",
        "slow": f"the GLM plus the logit offset {laws.SLOW_OFFSET:g} n^(-1/4) Z, a constructed "
        "error (E[Z | W1, W2] = 0)",
    },
    "treatment_learners": {
        "subset": "unpenalized logistic GLM on (W1, W2), declared by treatment_covariates",
        "intercept": "intercept only, declared by treatment_covariates",
        "full": "unpenalized logistic GLM on (W1, W2, Z), without the declaration",
    },
    "estimands": list(laws.ESTIMANDS),
    "g_bounds": list(laws.G_BOUNDS),
    "max_iter": laws.MAX_ITER,
    "tol": laws.TOL,
    "cross_fit": "False, except the stacked calibration cell (the shipped default folds)",
    "simultaneous_intervals": False,
    "settings": {name: vars(setting) for name, setting in laws.SETTINGS.items()},
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return laws.draw(scenario, n, seed), laws.truth(scenario)


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
    return cleverly_rows(laws.fit(frame, PRIMARY_SETTING), truth, scenario, replicate)


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
    truth = laws.truth(PRIMARY_SCENARIO)
    return {PRIMARY_SCENARIO: {name: truth[name] for name in PRIMARY_ESTIMANDS}}


__all__ = [
    "PROPERTY_CELLS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
