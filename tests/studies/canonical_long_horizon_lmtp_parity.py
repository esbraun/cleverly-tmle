"""``long-horizon-lmtp-parity``: ``mix`` against R ``lmtp`` 9996b04 at small n (X59 section 5).

``L_R`` at K = 5, m = 3 and n = 8,000, the plan ``mix`` alone, Wald-only, 500 replicates:

=========================  ==========================  =======================================
scenario                   subject                     comparator
=========================  ==========================  =======================================
``parity_tmle_mix``        ``LTMLE``, one fold         ``lmtp_tmle`` (9996b04 adapter)
``parity_sdr_mix``         ``SDR``, one fold           ``lmtp_sdr`` (shipped SDR adapter)
``parity_sdr_mix_folds``   ``SDR``, five folds         ``lmtp_sdr``, the realised folds
=========================  ==========================  =======================================

``mix`` is stochastic at every node, so every observed unit follows it and the pooled and the
stratified regressions coincide: G1 (1e-6 on estimate and SE) covers each scenario.
:mod:`tests.studies.long_horizon_parity` states what the record reads and what it does not
claim.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests import discrete_law_longitudinal_horizon as law_
from tests.parallel import STUDY_JOBS
from tests.studies import long_horizon_reference as reference_
from tests.studies.evidence import replication
from tests.studies.evidence.parity import ParityCell, ParityDeclaration, ReferenceSample
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.long_horizon_parity import PARITY_REPLICATES, run_parity, scenario_estimands
from tests.studies.long_horizon_primary import Scenario
from tests.studies.long_horizon_primary import draw_from_seed as _draw_from_seed

PRIMARY_REPLICATES = PARITY_REPLICATES
SEED = 20265911
RESAMPLING_SEED = 2026591101
SCENARIOS = (
    Scenario("parity_tmle_mix", 5, 3, law_.L_R, 8_000, ("mix",)),
    Scenario("parity_sdr_mix", 5, 3, law_.L_R, 8_000, ("mix",)),
    Scenario("parity_sdr_mix_folds", 5, 3, law_.L_R, 8_000, ("mix",), n_folds=5),
)
ESTIMATORS = {"parity_tmle_mix": "LTMLE", "parity_sdr_mix": "SDR", "parity_sdr_mix_folds": "SDR"}
COMPARATORS = {
    "parity_tmle_mix": "lmtp_tmle",
    "parity_sdr_mix": "lmtp_sdr",
    "parity_sdr_mix_folds": "lmtp_sdr",
}
PRIMARY_N = 8_000

STUDY = StudyRecord(
    name="long-horizon mix parity with R lmtp",
    slug="long-horizon-lmtp-parity",
    artifacts=ROOT / "tests" / "canonical" / "long_horizon_lmtp_parity",
    document="docs/technical-reference/method-evidence/long-horizon-longitudinal-tmle.md",
    anchor="long-horizon-longitudinal-tmle",
    scenarios=scenario_estimands(SCENARIOS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    implementation="cleverly-long-horizon-lmtp-parity",
    pending_reference="lmtp",
    pending_reason="X59 section 5: the R lmtp side runs reference-only after the subject phase",
    comparison="matching implementation",
    modules=(
        "tests/studies/canonical_long_horizon_lmtp_parity.py",
        "tests/studies/long_horizon_parity.py",
        "tests/studies/long_horizon_parity_properties.py",
        "tests/studies/long_horizon_reference.py",
        "tests/studies/long_horizon_primary.py",
        "tests/studies/long_horizon_common.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal_horizon.py",
        "tests/longitudinal_horizon_plans.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_long_horizon_lmtp_parity",
    properties_module="tests.studies.long_horizon_parity_properties",
    publication_policy="reporting",
    failure_bound={"cleverly-long-horizon-lmtp-parity": 0.01, "lmtp": 0.01},
)

CONFIGURATION: dict[str, Any] = {
    "estimator": ESTIMATORS,
    "comparator": COMPARATORS,
    "bootstrap": False,
    "outcome_learner": "QuasiBinomialGLM",
    "pseudo_learner": {"LTMLE": "QuasiBinomialGLM", "SDR": "LinearRegression"},
    "mechanism": "the law's, passed as probability columns",
    "laws": {s.name: [s.law.B, s.law.s, s.law.c0] for s in SCENARIOS},
    "n_folds": {s.name: s.n_folds for s in SCENARIOS},
}

#: Every margins pair is descriptive: the record reads its exact gates only.
COMPARATOR_DECLARATION = {
    "paired_role": "descriptive",
    # Every replicate is paired: the parity gates read each one, and the record's 500 were
    # chosen for them (section 4.2).
    "reference_replicates": dict.fromkeys(STUDY.scenarios, STUDY.replicates),
}
#: The comparator's declared environment: no mutation, and the declared learner (M2).
REFERENCE_ENVIRONMENT = reference_.LMTP_ENVIRONMENT
PARITY = ParityDeclaration(
    cells=tuple(ParityCell("G1", name, "ey_regimen[mix]") for name in ESTIMATORS)
)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """The subject phase: Wald-only ``LTMLE`` and ``SDR`` on every replicate."""
    return run_parity(STUDY, SCENARIOS, ESTIMATORS, replicates=replicates, n=n, n_jobs=n_jobs)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``scenario`` from ``seed``, with the exact truth of every estimand."""
    return _draw_from_seed(SCENARIOS, scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this record's own seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def reference_sample(scenario: str, n: int, replicate: int) -> ReferenceSample:
    """The R ``lmtp`` input of one replicate: its comparator follows the scenario."""
    return reference_.sample(STUDY, SCENARIOS, COMPARATORS[scenario], scenario, n, replicate)
