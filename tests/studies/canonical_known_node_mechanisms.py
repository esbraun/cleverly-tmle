"""Registered evidence for LTMLE on declared known node mechanisms.

The subject is ``LTMLE`` on a two-node SMART whose treatment and censoring factors the data
declares (``treatment_probabilities=`` and ``censoring_probabilities=``), so no mechanism
learner runs.  With every factor known the ICE-TMLE remainder is zero and the curve is
``D*(Qbar_inf, g0)`` (van der Laan and Gruber 2012, Theorem 2 and Section 4).  The law is the
SMART law of :mod:`tests.studies.known_mechanism_law`.

**The comparator** is R ``ltmle`` 1.3-0 with the declared factors as a numeric ``gform``,
``variance.method = "ic"`` and ``gbounds = c(1e-8, 1)``, through
``tests/canonical/known_node_mechanisms/run_study.R``.  The runner refuses a fit where a
cumulative bound binds.  Both sides fit the follower-stratified quasibinomial regressions of
the scenario's ``Qform``.

**The scenarios.**  ``smart_q_correct`` regresses ``Y`` on ``L0 + L1`` and ``L1`` on
``L0``, which is saturated for this law within each plan.  ``smart_q_wrong`` regresses ``Y``
on ``L0`` and ``L1`` on an intercept.  Each reports the means of ``always`` and ``never``
and their contrast.  The two scenarios read the same samples.

Publication policy is ``reporting``, with the red-cell route of
:mod:`tests.studies.canonical_known_mechanism` (owner ``X15-known-mechanism``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import known_mechanism_law as law
from tests.studies.canonical_ltmle import (
    LTMLE_SOURCE_COMMIT,
    LTMLE_TARBALL_SHA256,
    LTMLE_VERSION,
    R_BASE_IMAGE,
    regimen_initials,
    regimen_rows,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261152
RESAMPLING_SEED = 2026115201
IMPLEMENTATION = "cleverly-known-node-mechanisms"
REFERENCE = "ltmle-known-gform"
G_BOUNDS = (1e-8, 1.0)
PLANS: dict[str, tuple[int, int]] = {"always": (1, 1), "never": (0, 0)}
REGIMEN_REFERENCE = "never"
MEAN_NAMES = tuple(f"ey_regimen[{label}]" for label in PLANS)
CONTRAST_NAMES = ("ate_regimen[always vs never]",)
ESTIMANDS = (*MEAN_NAMES, *CONTRAST_NAMES)
Q_CORRECT = "smart_q_correct"
Q_WRONG = "smart_q_wrong"
SCENARIOS = (Q_CORRECT, Q_WRONG)
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["L0"],
    "time_varying": [[], ["L1"]],
    "censoring": ["C1", "C2"],
}
#: The declared factors, by column.
TREATMENT = {"A1": {0.0: "g1_0", 1.0: "g1_1"}, "A2": {0.0: "g2_0", 1.0: "g2_1"}}
CENSORING = {"C1": "r1", "C2": "r2"}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "known_mechanism_accuracy": (
        "ate_regimen__ltmle_known__q_wrong",
        "ate_regimen__ltmle_known_treatment__censoring_estimated",
    ),
}

STUDY = StudyRecord(
    name="LTMLE on declared known node mechanisms",
    slug="known-node-mechanisms",
    artifacts=ROOT / "tests" / "canonical" / "known_node_mechanisms",
    document="docs/technical-reference/method-evidence/known-treatment-mechanism.md",
    anchor="ltmle-on-declared-known-node-mechanisms",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={Q_WRONG: Q_CORRECT},
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    modules=(
        "tests/studies/canonical_known_node_mechanisms.py",
        "tests/studies/known_node_mechanisms_properties.py",
        "tests/studies/known_mechanism_law.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/ltmle/Dockerfile",
        "tests/canonical/known_node_mechanisms/run_study.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_known_node_mechanisms",
    properties_module="tests.studies.known_node_mechanisms_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "ltmle_version": LTMLE_VERSION,
    "ltmle_source_commit": LTMLE_SOURCE_COMMIT,
    "ltmle_tarball_sha256": LTMLE_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "ltmle(gform = <n x 4 declared factors>, Qform = <the scenario's>, stratify = TRUE, "
        "variance.method = 'ic', gbounds = c(1e-8, 1), SL.library = 'glm')"
    ),
}

CONFIGURATION = {
    "construction": "ordinary LTMLE dividing by the declared node factors",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "treatment_mechanism": "known at both nodes (declared by column)",
    "censoring_mechanism": "known at both nodes (declared by column)",
    "plans": {label: list(plan) for label, plan in PLANS.items()},
    "q_formulas": {
        Q_CORRECT: ["Q.kplus1 ~ L0", "Q.kplus1 ~ L0 + L1"],
        Q_WRONG: ["Q.kplus1 ~ 1", "Q.kplus1 ~ L0"],
    },
}


class ColumnQuasiBinomial(BaseEstimator):
    """:class:`QuasiBinomialGLM` on the first ``keep`` columns of each node's design.

    A node's outcome design is ``[L0]`` at node one and ``[L0, L1]`` at node two
    (:meth:`~cleverly.longitudinal.LongitudinalData.covariate_history`), so keeping one fewer
    column than the design has drops the latest covariate.
    """

    def __init__(self, drop_last: bool = False) -> None:
        self.drop_last = drop_last

    def _select(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        return values[:, :-1] if self.drop_last else values

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ColumnQuasiBinomial:
        selected = self._select(design)
        if selected.shape[1] == 0:
            selected = np.zeros((selected.shape[0], 0))
        self.model_ = QuasiBinomialGLM().fit(selected, target, sample_weight=sample_weight)
        return self

    def predict(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._select(design)), dtype=float)


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    outcome: str = "wrong",
    censoring: str = "known",
    treatment_learner: Any = None,
    censoring_learner: Any = None,
) -> Any:
    """Fit this study's LTMLE on one sample.

    ``censoring="known"`` declares the retention probabilities; ``"estimated"`` fits
    ``censoring_learner`` instead.
    """
    learner = ColumnQuasiBinomial(drop_last=outcome == "wrong")
    declared: dict[str, Any] = {"treatment_probabilities": TREATMENT}
    if censoring == "known":
        declared["censoring_probabilities"] = CENSORING
    return LTMLE(
        {label: tuple(plan) for label, plan in PLANS.items()},
        reference=REGIMEN_REFERENCE,
        outcome_learner=learner,
        pseudo_learner=learner,
        treatment_learner=treatment_learner,
        censoring_learner=censoring_learner,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, **COLUMNS, **declared)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIOS:
        raise KeyError(scenario)
    return law.smart_sample(n, seed), truth()


def truth() -> dict[str, float]:
    """Every reported parameter, by exact enumeration."""
    means = law.smart_truth(PLANS)
    out = {f"ey_regimen[{label}]": value for label, value in means.items()}
    out["ate_regimen[always vs never]"] = means["always"] - means["never"]
    return out


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, reference_truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, outcome="correct" if scenario == Q_CORRECT else "wrong")
    return regimen_rows(
        STUDY,
        result,
        reference_truth,
        regimen_initials(result, PLANS, CONTRAST_NAMES),
        ESTIMANDS,
        scenario,
        replicate,
        n=len(frame),
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference_truth = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference_truth.items()
    ]
    return sample, truths, cleverly_rows(frame, reference_truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((scenario, replicate, n),) for scenario in SCENARIOS for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]


def plan_names() -> Sequence[str]:
    """The plan labels, in report order."""
    return tuple(PLANS)
