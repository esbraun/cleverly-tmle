r"""Long-horizon evidence for ``LTMLE`` at three to five three-level decisions.

The subject is :class:`~cleverly.longitudinal.LTMLE` with three-level nodes on the law family of
:mod:`tests.discrete_law_longitudinal_horizon`, ordinary and five-fold cross-fitted.  A scenario
whose rows fail the exact-law criterion on L_R splits by law: each fallback law a row needs gets
one companion scenario, which fits the reference and the plans of the rows it gates.

=============================  =================================================================
scenario                       subject
=============================  =================================================================
``categorical_k3``             L_R, n = 28,000: ``low``, ``high``, ``respond``, ``switch``,
                               ``mix``, ``mixlast``
``categorical_k4``             L_R, n = 128,000: ``low``, ``respond``, ``switch``, ``mix``,
                               ``mixlast``
``categorical_k4_high``        (0.75, 0.1, 3.5), n = 108,000: ``low``, ``high``
``categorical_k4_static``      (1, 0, 4.5), n = 124,000: ``low``, ``high``
``categorical_k5``             L_R, n = 116,000: ``low``, ``respond``, ``mix``, ``mixlast``
``categorical_k5_static``      (0.5, 0.05, 3.5), n = 124,000: ``low``, ``high``, ``switch``
``categorical_k5_switch``      (0.5, 0.1, 3.5), n = 72,000: ``low``, ``switch``
``categorical_k5_high``        (0.25, 0, 3.5), n = 64,000: ``low``, ``high``
``categorical_k5_mix``         (1, 0.05, 3.5), n = 96,000: ``low``, ``mix``
``categorical_k5_crossfit``    ``categorical_k5`` at five folds, on its realized sample
=============================  =================================================================

A law is written ``(B, s, c0)``.  The policy scenarios also report ``mix`` against ``mixlast``,
the paired contrast whose two plans share every follower path.

**Comparator.**  R ``lmtp`` is the declared comparator, and its runs are pending: the record
declares ``pending_reference="lmtp"`` and keeps ``reference=None``.  The primary phase writes no
sample.  ``tests/canonical/long_horizon_categorical_ltmle/run_study.R`` is a stub that names the
owning item.
"""

from __future__ import annotations

from tests import discrete_law_longitudinal_horizon as law_
from tests.discrete_law_longitudinal_horizon import Law
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.long_horizon_common import G_BOUNDS
from tests.studies.long_horizon_primary import Scenario, declared, run_primary

#: Provisional: the truth-row rule of the design picks R from 1,200, 2,000 and 3,000.
PRIMARY_REPLICATES = 1_200
SEED = 20263911
RESAMPLING_SEED = 2026391101
ESTIMATOR = "LTMLE"

#: The outer fold count of the cross-fitted scenario.
N_FOLDS = 5
#: The K = 5 plans on L_R, reference first.
K5_PLANS = ("low", "respond", "mix", "mixlast")
#: The policy and its last-node stop, which share every follower path.
PAIRED = (("mix", "mixlast"),)

SCENARIOS = (
    Scenario(
        "categorical_k3",
        3,
        3,
        law_.L_R,
        28_000,
        ("low", "high", "respond", "switch", "mix", "mixlast"),
        paired=PAIRED,
    ),
    Scenario(
        "categorical_k4",
        4,
        3,
        law_.L_R,
        128_000,
        ("low", "respond", "switch", "mix", "mixlast"),
        paired=PAIRED,
    ),
    Scenario("categorical_k4_high", 4, 3, Law(0.75, 0.1, 3.5), 108_000, ("low", "high")),
    Scenario("categorical_k4_static", 4, 3, Law(1.0, 0.0, 4.5), 124_000, ("low", "high")),
    Scenario("categorical_k5", 5, 3, law_.L_R, 116_000, K5_PLANS, paired=PAIRED),
    Scenario(
        "categorical_k5_static", 5, 3, Law(0.5, 0.05, 3.5), 124_000, ("low", "high", "switch")
    ),
    Scenario("categorical_k5_switch", 5, 3, Law(0.5, 0.1, 3.5), 72_000, ("low", "switch")),
    Scenario("categorical_k5_high", 5, 3, Law(0.25, 0.0, 3.5), 64_000, ("low", "high")),
    Scenario("categorical_k5_mix", 5, 3, Law(1.0, 0.05, 3.5), 96_000, ("low", "mix")),
    Scenario(
        "categorical_k5_crossfit",
        5,
        3,
        law_.L_R,
        116_000,
        K5_PLANS,
        n_folds=N_FOLDS,
        paired=PAIRED,
    ),
)
SCENARIO_ESTIMANDS, SCENARIO_N = declared(SCENARIOS)
PRIMARY_N = min(SCENARIO_N.values())

STUDY = StudyRecord(
    name="long-horizon categorical longitudinal TMLE",
    slug="long-horizon-categorical-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "long_horizon_categorical_ltmle",
    document="docs/technical-reference/method-evidence/long-horizon-categorical-longitudinal-tmle.md",
    anchor="long-horizon-categorical-longitudinal-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    scenario_seed_owners={"categorical_k5_crossfit": "categorical_k5"},
    scenario_n=SCENARIO_N,
    resampling_seed=RESAMPLING_SEED,
    implementation="cleverly-long-horizon-categorical-ltmle",
    pending_reference="lmtp",
    pending_reason=(
        "X59: R lmtp comparator runs deferred by the owner on 2026-10-14; X39 accepted on Python property evidence"
    ),
    comparison="matching implementation",
    comparison_overrides={"categorical_k5_crossfit:*": "different estimator"},
    modules=(
        "tests/studies/canonical_long_horizon_categorical_ltmle.py",
        "tests/studies/long_horizon_categorical_ltmle_properties.py",
        "tests/studies/long_horizon_properties.py",
        "tests/studies/canonical_long_horizon_ltmle.py",
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
    runner_module="tests.studies.canonical_long_horizon_categorical_ltmle",
    properties_module="tests.studies.long_horizon_categorical_ltmle_properties",
    publication_policy="reporting",
    failure_bound={"cleverly-long-horizon-categorical-ltmle": 0.01, "lmtp": 0.01},
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
