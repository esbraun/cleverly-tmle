"""``long-horizon-ltmle-parity``: ``LTMLE`` against R ``ltmle`` 1.3-0 at small n (X59 section 5).

One scenario, ``parity_binary_k5``: ``L_R`` at K = 5, m = 2 and n = 4,000, the four
deterministic plans of ``long-horizon-ltmle`` and their contrasts against ``never``, Wald-only,
500 replicates.  :mod:`tests.studies.long_horizon_parity` states what the record reads and what
it does not claim.  G-P and G-C cover every estimand; both tolerances are declared from the
smoke maxima by the declaration commit, as for the primary study.
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
SEED = 20265901
RESAMPLING_SEED = 2026590101
SCENARIOS = (
    Scenario("parity_binary_k5", 5, 2, law_.L_R, 4_000, ("never", "always", "respond", "switch")),
)
ESTIMATORS = {"parity_binary_k5": "LTMLE"}
PRIMARY_N = 4_000

STUDY = StudyRecord(
    name="long-horizon LTMLE parity with R ltmle",
    slug="long-horizon-ltmle-parity",
    artifacts=ROOT / "tests" / "canonical" / "long_horizon_ltmle_parity",
    document="docs/technical-reference/method-evidence/long-horizon-longitudinal-tmle.md",
    anchor="long-horizon-longitudinal-tmle",
    scenarios=scenario_estimands(SCENARIOS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    implementation="cleverly-long-horizon-ltmle-parity",
    pending_reference="ltmle",
    pending_reason="X59 section 5: the R ltmle side runs reference-only after the subject phase",
    comparison="matching implementation",
    modules=(
        "tests/studies/canonical_long_horizon_ltmle_parity.py",
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
    runner_module="tests.studies.canonical_long_horizon_ltmle_parity",
    properties_module="tests.studies.long_horizon_parity_properties",
    publication_policy="reporting",
    failure_bound={"cleverly-long-horizon-ltmle-parity": 0.01, "ltmle": 0.01},
)

CONFIGURATION: dict[str, Any] = {
    "estimator": ESTIMATORS,
    "bootstrap": False,
    "outcome_learner": "QuasiBinomialGLM",
    "mechanism": "the law's, passed as probability columns",
    "laws": {s.name: [s.law.B, s.law.s, s.law.c0] for s in SCENARIOS},
    "plans": {s.name: list(s.plans) for s in SCENARIOS},
}

#: Every margins pair is descriptive: the record reads its exact gates only.
COMPARATOR_DECLARATION = {"paired_role": "descriptive"}
PARITY_SCENARIOS = frozenset(ESTIMATORS)
PARITY = ParityDeclaration(
    cells=tuple(
        ParityCell(gate, name, estimand)
        for gate in ("G-P", "G-C")
        for name in sorted(PARITY_SCENARIOS)
        for estimand in STUDY.scenarios[name]
    )
)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """The subject phase: Wald-only ``LTMLE`` on every replicate."""
    return run_parity(STUDY, SCENARIOS, ESTIMATORS, replicates=replicates, n=n, n_jobs=n_jobs)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``scenario`` from ``seed``, with the exact truth of every estimand."""
    return _draw_from_seed(SCENARIOS, scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this record's own seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def reference_sample(scenario: str, n: int, replicate: int) -> ReferenceSample:
    """The R ``ltmle`` input of one replicate, from this record's seed stream."""
    return reference_.sample(
        STUDY, SCENARIOS, "ltmle", scenario, n, replicate, parity_scenarios=PARITY_SCENARIOS
    )
