r"""Declared evidence for the known-score ATT and ATC on point TMLE (``known-score-att``).

The subject is ``TMLE(conditional_form="known_score")`` on data that declares its treatment
mechanism (``treatment_probabilities=``): the ATT is :math:`P_n\{g_0\tau^*\}/P_n g_0` with the
curve :math:`D_{KS}`, the canonical gradient of the model in which :math:`g_0` is known, and
the ATC weights by :math:`1-g_0` (X35 plan sections 1 and 2, rev 2.1).  Laws, learners and
truths are :mod:`tests.studies.known_score_att_law`; the property families are
:mod:`tests.studies.known_score_att_properties`:

==========================  =================================================================
family                      cells
==========================  =================================================================
``interval_calibration``    Law S, correct outcome regression, n = 2,000: the known-score ATT
                            and ATC in sample and stacked over 5 folds (``q_bounds=(-8, 8)``),
                            and the arm-mean ATT and ATC in sample on the same draws, each
                            with its exact efficiency bound and the derived
                            ``shrunken_se_control`` and ``noise_control``
``efficiency_gain``         the paired spread ratio SD(known score) / SD(arm mean) of the ATT
                            and of the ATC on the calibration draws (positives), and on Law
                            S0, where the gain is zero (controls)
``known_score_accuracy``    the known-score ATT and ATC with a wrong outcome regression on Law
                            S and on Law B (two arms), and ATT[1], ATT[2], ATC[1], ATC[2] on
                            three-arm Law B; the control declares the constant mechanism 0.49
                            on Law S
``root_n_and_efficiency``   the known-score ATT on Law S, correct outcome regression, at
                            n = 500, 2,000 and 8,000, with the two ``root_n_rate`` slopes
==========================  =================================================================

**No comparator.**  No maintained package reports the known-score ATT with its own curve
(plan section 1, "Out of scope").  Every cell reads its exact truth.

**Publication policy** is ``reporting``.  The red-cell reading is declared in
``tests/canonical/known_score_att/regenerate.py`` before any run.

**Registered.**  The declared run is committed in ``tests/canonical/known_score_att/``; support
rows ``T55`` cite its cells.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import known_score_att_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import primary_rows

SLUG = "known-score-att"
SEED = 20263501
RESAMPLING_SEED = 20263502
FAILURE_BOUND = 0.01
IMPLEMENTATION = "cleverly"

#: The primary phase: the in-sample known-score fit on Law S at n = 2,000, correct outcome
#: regression.
PRIMARY_SCENARIO = "S"
PRIMARY_SETTING = "s_known_score"
PRIMARY_N = 2_000
#: The primary rule (:func:`tests.studies.repeated_sizing.primary_replicates`, one
#: implementation) at the committed 8,000-draw calibration pilot; the design test recomputes it.
PRIMARY_REPLICATES = 2_400
PRIMARY_ESTIMANDS = ("att", "atc")

#: The calibration labels: ``<estimand>_<form>_<splitting>``.
CALIBRATION_LABELS = (
    "att_known_score_in_sample",
    "atc_known_score_in_sample",
    "att_known_score_stacked",
    "atc_known_score_stacked",
    "att_arm_mean_in_sample",
    "atc_arm_mean_in_sample",
)
#: The ``efficiency_gain`` pairs: the numerator cell (known score) and the denominator cell
#: (arm mean) of each label.  ``att`` and ``atc`` read the calibration draws of Law S, and the
#: ``zero_gain`` labels read Law S0.
GAIN_LABELS = ("att", "atc", "att_zero_gain", "atc_zero_gain")
#: The ``known_score_accuracy`` cells.
ACCURACY_CELLS = (
    "s_att__q_wrong",
    "s_atc__q_wrong",
    "b2_att__q_wrong",
    "b2_atc__q_wrong",
    "b3_att_1__q_wrong",
    "b3_att_2__q_wrong",
    "b3_atc_1__q_wrong",
    "b3_atc_2__q_wrong",
    "s_att__declared_constant",
)
RATE_SIZES = (500, 2_000, 8_000)

#: Filled by :mod:`tests.studies.known_score_att_properties`, which this module cannot import;
#: the design test pins the two equal.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "efficiency_gain": tuple(
        f"{label}__{form}" for label in GAIN_LABELS for form in ("known_score", "arm_mean")
    ),
    "known_score_accuracy": ACCURACY_CELLS,
    "root_n_and_efficiency": tuple(f"n_{size}" for size in RATE_SIZES),
    "root_n_rate": ("empirical_sd", "reported_se"),
}

STUDY = StudyRecord(
    name="known-score ATT and ATC on point TMLE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "known_score_att",
    document="docs/technical-reference/method-evidence/known-score-att.md",
    anchor="known-score-att-and-atc-on-point-tmle",
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
        "tests/studies/known_score_att.py",
        "tests/studies/known_score_att_properties.py",
        "tests/studies/known_score_att_law.py",
        "tests/studies/known_mechanism_law.py",
        "tests/studies/point_study_helpers.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/fluctuation/iterative.py",
        "src/cleverly/inference/influence.py",
        "src/cleverly/targets/builtin.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_failures.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.known_score_att",
    properties_module="tests.studies.known_score_att_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds={
        "att": law.efficiency_sd("S", "att", "known_score"),
        "atc": law.efficiency_sd("S", "atc", "known_score"),
    },
    calibration_efficiency_ratio=True,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "point TMLE with a declared treatment mechanism and "
    'conditional_form="known_score" (g-weighted ATT and ATC plug-in, curve D_KS)',
    "law_S": "W1 ~ U(-sqrt3, sqrt3), W2 ~ Bernoulli(0.5); g0 = expit(-0.3 + W1 + 0.5 W2); "
    "Y = W1 + 0.5 W2 + A (1 + 1.5 W1) + eps, eps ~ U(-sqrt3, sqrt3)",
    "law_S0": "law S with tau = 1 (zero known-score gain)",
    "law_B": "tests/studies/known_mechanism_law.py, two and three arms",
    "outcome_learners": {
        "correct": "least squares on (1, A, W1, W2, A W1) (law S)",
        "wrong": "least squares on (1, A, W2) (law S); main-terms logistic on (A, W1, W3) (law B)",
    },
    "treatment_mechanism": "declared by column (p0, p1[, p2]); no treatment learner runs",
    "control_declared_constant": law.CONTROL_CONSTANT,
    "estimands": list(PRIMARY_ESTIMANDS),
    "g_bounds": "default",
    "q_bounds": f"declared {list(law.Q_BOUNDS)} on the stacked cells only",
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
    exact = law.truth(scenario)
    return law.draw(scenario, n, seed), {name: exact[name] for name in PRIMARY_ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def known_score_initials(result: Any) -> dict[str, float]:
    """The untargeted known-score ATT and ATC: ``sum g tau_n / sum g`` with the declared ``g``."""
    weights = np.asarray(result.data.weights, dtype=float)
    blip = np.asarray(
        result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[1.0]), dtype=float
    ) - np.asarray(
        result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[0.0]), dtype=float
    )
    declared = np.asarray(result.data.known_treatment.values, dtype=float)
    treated = declared[:, list(result.data.arm_codes).index(1.0)]
    return {
        "att": float(np.sum(weights * treated * blip) / np.sum(weights * treated)),
        "atc": float(np.sum(weights * (1.0 - treated) * blip) / np.sum(weights * (1.0 - treated))),
    }


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
        initials=known_score_initials(result),
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
    exact = law.truth(PRIMARY_SCENARIO)
    return {PRIMARY_SCENARIO: {name: exact[name] for name in PRIMARY_ESTIMANDS}}


__all__ = [
    "CALIBRATION_LABELS",
    "PROPERTY_CELLS",
    "STUDY",
    "draw_and_fit",
    "draw_scenario",
    "truth_table",
]
