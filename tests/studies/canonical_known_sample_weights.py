r"""Declared evidence for ``mechanism_law="sample"`` (``known-mechanism-sample-weights``, Fix-16).

The subject is ``TMLE`` and ``LTMLE`` fitted on weighted data that declares the treatment
mechanism of the rows as drawn (``treatment_probabilities=``) with ``mechanism_law="sample"``:
the fit estimates the mechanism of the weight-tilted law as :math:`\hat g_w \propto g_0 \hat r`,
with :math:`\hat r` the treatment learner's classification of the treatment with sample weights
:math:`w / g_0` (Fix-16 plan sections 2.2 and 3).  It is an estimated-mechanism fit, so its
interval needs the outcome regression and the tilt classifier both consistent, at a product
rate.  The laws, learners and exact-law quantities are
:mod:`tests.studies.known_sample_weights_law`; the property families are
:mod:`tests.studies.known_sample_weights_properties`:

==================================  ==========================================================
family                              cells
==================================  ==========================================================
``interval_calibration``            law SW, ``"sample"`` with the correct outcome regression
                                    and the correct main-terms tilt classifier, five-fold
                                    cross-fit, ATE, at n = 1,000, 4,000 and 16,000; law LW,
                                    LTMLE ``"sample"`` with correct regressions and tilt
                                    classifiers, five-fold cross-fit with declared outcome
                                    bounds, the regimen contrast at n = 4,000: 4 gated
                                    positives, each with the derived ``shrunken_se_control``
                                    and ``noise_control``
``sample_bias``                     the same fits: gated positives, bias inside the margin
``target_reading_control``          ``"target"`` (the old reading, dividing by the
                                    declaration) with the main-terms outcome regression (SW)
                                    or an intercept-only node-one regression (LW), on the
                                    draws of each ``sample_bias`` cell: gated controls, bias
                                    beyond the margin
``sample_misspecified``             ``"sample"`` with the same misspecified regression, on the
                                    same draws: gated positives, bias inside the margin, the
                                    direct witness of the tilt (calibration published,
                                    ungated)
``tilt_misspecified``               law SW at n = 4,000, ``"sample"`` with the correct outcome
                                    regression and a tilt classifier that drops ``W1``: a
                                    gated positive, bias inside the margin
``both_misspecified``               the same wrong tilt with the main-terms outcome
                                    regression: a gated control, bias beyond the margin
==================================  ==========================================================

**No comparator.**  No external package implements the tilted mechanism from a declaration;
the fast tier's exact-law tests (``tests/unit/test_known_mechanism_weights.py`` and its
longitudinal sibling) pin the construction.  The study reads every cell against its exact truth.

**Publication policy** is ``gated``.  The red-cell reading is declared in
``tests/canonical/known_sample_weights/regenerate.py`` before any run.

**Registered.**  The declared run at ``26b53828`` passed every gated verdict (3 of 3 primary
rows, 26 of 26 property cells, no failed fit); its artifacts are
``tests/canonical/known_sample_weights`` and its support rows ``T55`` and ``G62``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import known_sample_weights_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import initial_estimates, primary_rows

SLUG = "known-mechanism-sample-weights"
SEED = 20261641
RESAMPLING_SEED = 20261642
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the gated point fit on law SW at n = 4,000, its ATE and arm means, on
#: this record's own seed (the property cells read their own streams).
PRIMARY_SCENARIO = "SW"
PRIMARY_SETTING = "sample_correct"
PRIMARY_N = 4_000
PRIMARY_REPLICATES = 2_000
PRIMARY_ESTIMANDS = law.ESTIMANDS

#: The point sizes and the longitudinal size of the property cells.
POINT_SIZES = (1_000, 4_000, 16_000)
LONGITUDINAL_N = 4_000
POINT_LABELS = tuple(f"sw_n{n}" for n in POINT_SIZES)
LONGITUDINAL_LABEL = f"lw_n{LONGITUDINAL_N}"
CALIBRATION_LABELS = (*POINT_LABELS, LONGITUDINAL_LABEL)

#: Filled by :mod:`tests.studies.known_sample_weights_properties`, which this module cannot
#: import; the design test pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "sample_bias": CALIBRATION_LABELS,
    "target_reading_control": CALIBRATION_LABELS,
    "sample_misspecified": CALIBRATION_LABELS,
    "tilt_misspecified": ("sw_n4000",),
    "both_misspecified": ("sw_n4000",),
}

STUDY = StudyRecord(
    name="A declared treatment mechanism beside sample-reading weights",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "known_sample_weights",
    document="docs/technical-reference/method-evidence/known-mechanism-sample-weights.md",
    anchor="a-declared-treatment-mechanism-beside-sample-reading-weights",
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
        "tests/studies/canonical_known_sample_weights.py",
        "tests/studies/known_sample_weights_properties.py",
        "tests/studies/known_sample_weights_law.py",
        "tests/studies/point_study_helpers.py",
        "src/cleverly/data/known_mechanism.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/longitudinal/sequential.py",
        "src/cleverly/longitudinal/estimator.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_known_sample_weights",
    properties_module="tests.studies.known_sample_weights_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": (
        "point TMLE and LTMLE on weighted data with treatment_probabilities= and "
        "mechanism_law='sample' (tilted mechanism g0 r / sum g0 r, r the treatment learner fitted "
        "with sample weights w / g0); the control reads mechanism_law='target'"
    ),
    "law_SW": (
        "W1 ~ N(0, 1), W2 ~ Bern(0.5); g0(1 | W) = 0.3 + 0.4 W2 (declared); "
        "Y ~ Bern(expit(-1 + A + W1 + 1.5 A W1 - W1^2 / 2 + 0.5 W2)); "
        "w = exp(0.4 A + 0.5 A W1 + 0.15 W1), so r(1 | W) = expit(0.4 + 0.5 W1)"
    ),
    "law_LW": (
        "L0 ~ N(0, 1); A1 ~ Bern(expit(0.4 L0)) (declared); L1 ~ N(0.5 L0 + 0.5 A1, 1); "
        "A2 ~ Bern(expit(-0.2 + 0.6 L1)) (declared); "
        "Y = 0.5 L0 + 0.4 A1 + 0.5 A1 L0 + 0.6 L1 + 0.5 A2 + 0.3 A1 A2 + N(0, 1); "
        "w = exp(0.4 A1 + 0.5 A1 L0 + 0.15 L0)"
    ),
    "outcome_learners": {
        "correct": "unpenalized logistic GLM on (A, W1, A W1, W1^2, W2), by Newton",
        "main": "unpenalized logistic GLM on (A, W1, W2), by Newton (misspecified)",
        "LW": "weighted least squares on each node's follower design (correct)",
        "LW_flat": "the weighted mean at node one (misspecified), least squares at node two",
    },
    "tilt_classifier": {
        "correct": "unpenalized main-terms logistic GLM on the node's design, by Newton",
        "w2": "unpenalized logistic GLM on W2 alone, by Newton (misspecified: drops W1)",
    },
    "point_fits": {name: vars(setting) for name, setting in law.SETTINGS.items()},
    "ltmle_fits": {name: list(setting) for name, setting in law.LW_SETTINGS.items()},
    "point_cross_fit": f"{law.N_FOLDS} folds",
    "ltmle_folds": law.LW_FOLDS,
    "ltmle_q_bounds": list(law.LW_Q_BOUNDS),
    "estimands": list(law.ESTIMANDS),
    "ltmle_estimands": list(law.LONGITUDINAL_ESTIMANDS),
    "g_bounds": list(law.G_BOUNDS),
    "ltmle_g_bounds": list(law.LW_G_BOUNDS),
    "max_iter": law.MAX_ITER,
    "tol": law.TOL,
    "simultaneous_intervals": False,
}


# ---------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in STUDY.scenarios:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(n, seed), law.truth()


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
    truth = law.truth()
    return {PRIMARY_SCENARIO: {name: truth[name] for name in PRIMARY_ESTIMANDS}}


__all__ = [
    "PROPERTY_CELLS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
