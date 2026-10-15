r"""Long-horizon evidence for ordinary ``LTMLE`` at three to five binary decisions.

The subject is :class:`~cleverly.longitudinal.LTMLE` with binary nodes on the law family of
:mod:`tests.discrete_law_longitudinal_horizon`, at K = 3, 4 and 5, with a duration MSM scenario.

===================  ===========================================================================
scenario             subject
===================  ===========================================================================
``binary_k3``        L_R, n = 4,000: ``never``, ``always``, ``respond``, ``switch``
``binary_k4``        L_R, n = 16,000: the same plans
``binary_k5``        L_R, n = 76,000: the same plans, ``coin`` and ``coinlast``, and the paired
                     contrast ``coin`` against ``coinlast``
``binary_k5_msm``    L_R, n = 48,000: ``dur0`` to ``dur5``, the working model intercept +
                     duration (identity link, unit weights); its two coefficients only
===================  ===========================================================================

Each scenario runs at the largest exact-law tail size :math:`n_\kappa` of the rows it gates,
rounded up to 4,000 (``tests/unit/test_long_horizon_design.py``).  The mechanism is the law's,
passed as probability columns; the outcome regressions are the declared quasibinomial GLM.

**Comparator.**  R ``ltmle`` 1.3-0 is the declared comparator, and its runs are pending: the record
declares ``pending_reference="ltmle"`` and keeps ``reference=None``.  The primary phase writes no
sample.  ``tests/canonical/long_horizon_ltmle/run_study.R`` is a stub that names the owning item.
"""

from __future__ import annotations

from tests import discrete_law_longitudinal_horizon as law_
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.long_horizon_common import G_BOUNDS
from tests.studies.long_horizon_primary import Scenario, declared, run_primary

#: Provisional: the truth-row rule of the design picks R from 1,200, 2,000 and 3,000.
PRIMARY_REPLICATES = 1_200
SEED = 20263901
RESAMPLING_SEED = 2026390101
ESTIMATOR = "LTMLE"

#: The binary plans of every K, reference first.
BINARY = ("never", "always", "respond", "switch")

SCENARIOS = (
    Scenario("binary_k3", 3, 2, law_.L_R, 4_000, BINARY),
    Scenario("binary_k4", 4, 2, law_.L_R, 16_000, BINARY),
    Scenario(
        "binary_k5",
        5,
        2,
        law_.L_R,
        76_000,
        (*BINARY, "coin", "coinlast"),
        paired=(("coin", "coinlast"),),
    ),
    Scenario(
        "binary_k5_msm",
        5,
        2,
        law_.L_R,
        48_000,
        tuple(f"dur{j}" for j in range(6)),
        msm=True,
    ),
)
SCENARIO_ESTIMANDS, SCENARIO_N = declared(SCENARIOS)
PRIMARY_N = min(SCENARIO_N.values())

STUDY = StudyRecord(
    name="long-horizon longitudinal TMLE",
    slug="long-horizon-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "long_horizon_ltmle",
    document="docs/technical-reference/method-evidence/long-horizon-longitudinal-tmle.md",
    anchor="long-horizon-longitudinal-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    scenario_seed_owners={},
    scenario_n=SCENARIO_N,
    resampling_seed=RESAMPLING_SEED,
    implementation="cleverly-long-horizon-ltmle",
    pending_reference="ltmle",
    pending_reason=(
        "X59: R ltmle 1.3-0 comparator runs deferred by the owner on 2026-10-14; X39 accepted on Python property evidence"
    ),
    comparison="matching implementation",
    modules=(
        "tests/studies/canonical_long_horizon_ltmle.py",
        "tests/studies/long_horizon_ltmle_properties.py",
        "tests/studies/long_horizon_properties.py",
        "tests/studies/canonical_long_horizon_categorical_ltmle.py",
        "tests/studies/canonical_long_horizon_sdr.py",
        "tests/long_horizon_design.py",
        "tests/long_horizon_sampling_sd.py",
        "tests/studies/long_horizon_primary.py",
        "tests/studies/long_horizon_common.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal_horizon.py",
        "tests/longitudinal_horizon_plans.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_long_horizon_ltmle",
    properties_module="tests.studies.long_horizon_ltmle_properties",
    publication_policy="reporting",
    failure_bound={"cleverly-long-horizon-ltmle": 0.01, "ltmle": 0.01},
)

CONFIGURATION = {
    "estimator": ESTIMATOR,
    "n_folds": {scenario.name: scenario.n_folds for scenario in SCENARIOS},
    "g_bounds": list(G_BOUNDS),
    "outcome_learner": "QuasiBinomialGLM",
    "pseudo_learner": "QuasiBinomialGLM",
    "mechanism": "the law's, passed as probability columns",
    "laws": {
        scenario.name: [scenario.law.B, scenario.law.s, scenario.law.c0] for scenario in SCENARIOS
    },
    "plans": {scenario.name: list(scenario.plans) for scenario in SCENARIOS},
}


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """The primary phase: every scenario at its declared size, or every one at an override ``n``."""
    return run_primary(STUDY, SCENARIOS, ESTIMATOR, replicates=replicates, n=n, n_jobs=n_jobs)
