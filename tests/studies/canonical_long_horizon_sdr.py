r"""Long-horizon evidence for ``SDR`` at three to five decisions, binary and three-level.

The subject is :class:`~cleverly.longitudinal.SDR` on the law family of
:mod:`tests.discrete_law_longitudinal_horizon`, ordinary and five-fold cross-fitted.  The
regression at the horizon is the declared quasibinomial GLM and every earlier one is least
squares on the transform, unclipped.

==================================  ============================================================
scenario                            subject
==================================  ============================================================
``sdr_binary_k3``, ``_k4``, ``_k5``  L_R, n = 4,000, 16,000, 76,000: ``never``, ``always``,
                                    ``respond``, ``switch``, ``coin``
``sdr_categorical_k5``              L_R, n = 116,000: ``low``, ``respond``, ``mix``, ``mixlast``
``sdr_categorical_k5_crossfit``     the same at five folds, on its realized sample
``sdr_categorical_k5_static``       (0.5, 0.05, 3.5), n = 124,000: ``low``, ``high``, ``switch``
``sdr_categorical_k5_switch``       (0.5, 0.1, 3.5), n = 72,000: ``low``, ``switch``
``sdr_categorical_k5_high``         (0.25, 0, 3.5), n = 64,000: ``low``, ``high``
``sdr_categorical_k5_mix``          (1, 0.05, 3.5), n = 96,000: ``low``, ``mix``
``sdr_policy_k5``                   L_R, n = 48,000: ``low``, ``mix``, ``taper``
``sdr_policy_k5_mix``               (1, 0.05, 3.5), n = 96,000: ``low``, ``mix``, ``taper``
``sdr_policy_k5_taper``             (1, 0, 3.5), n = 76,000: ``low``, ``taper``
==================================  ============================================================

The policy scenarios also report ``taper`` against ``mix``, and the categorical ones ``mix``
against ``mixlast``.

**Comparator.**  R ``lmtp`` is the declared comparator, and its runs are pending: the record
declares ``pending_reference="lmtp"`` and keeps ``reference=None``.  The primary phase writes no
sample.  ``tests/canonical/long_horizon_sdr/run_study.R`` is a stub that names the owning item.
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
SEED = 20263921
RESAMPLING_SEED = 2026392101
ESTIMATOR = "SDR"

#: The outer fold count of the cross-fitted scenario.
N_FOLDS = 5
#: The binary plans, reference first.
BINARY = ("never", "always", "respond", "switch", "coin")
#: The K = 5 three-level plans on L_R, reference first.
K5_PLANS = ("low", "respond", "mix", "mixlast")
#: The policy plans, reference first.
POLICY = ("low", "mix", "taper")
#: The policy and its last-node stop, which share every follower path.
PAIRED = (("mix", "mixlast"),)
#: The tapered policy against the policy it tapers.
TAPER = (("taper", "mix"),)

SCENARIOS = (
    Scenario("sdr_binary_k3", 3, 2, law_.L_R, 4_000, BINARY),
    Scenario("sdr_binary_k4", 4, 2, law_.L_R, 16_000, BINARY),
    Scenario("sdr_binary_k5", 5, 2, law_.L_R, 76_000, BINARY),
    Scenario("sdr_categorical_k5", 5, 3, law_.L_R, 116_000, K5_PLANS, paired=PAIRED),
    Scenario(
        "sdr_categorical_k5_crossfit",
        5,
        3,
        law_.L_R,
        116_000,
        K5_PLANS,
        n_folds=N_FOLDS,
        paired=PAIRED,
    ),
    Scenario(
        "sdr_categorical_k5_static",
        5,
        3,
        Law(0.5, 0.05, 3.5),
        124_000,
        ("low", "high", "switch"),
    ),
    Scenario("sdr_categorical_k5_switch", 5, 3, Law(0.5, 0.1, 3.5), 72_000, ("low", "switch")),
    Scenario("sdr_categorical_k5_high", 5, 3, Law(0.25, 0.0, 3.5), 64_000, ("low", "high")),
    Scenario("sdr_categorical_k5_mix", 5, 3, Law(1.0, 0.05, 3.5), 96_000, ("low", "mix")),
    Scenario("sdr_policy_k5", 5, 3, law_.L_R, 48_000, POLICY, paired=TAPER),
    Scenario("sdr_policy_k5_mix", 5, 3, Law(1.0, 0.05, 3.5), 96_000, POLICY, paired=TAPER),
    Scenario("sdr_policy_k5_taper", 5, 3, Law(1.0, 0.0, 3.5), 76_000, ("low", "taper")),
)
SCENARIO_ESTIMANDS, SCENARIO_N = declared(SCENARIOS)
PRIMARY_N = min(SCENARIO_N.values())

STUDY = StudyRecord(
    name="long-horizon sequentially doubly robust estimation",
    slug="long-horizon-sdr",
    artifacts=ROOT / "tests" / "canonical" / "long_horizon_sdr",
    document="docs/technical-reference/method-evidence/long-horizon-sequentially-doubly-robust-estimation.md",
    anchor="long-horizon-sequentially-doubly-robust-estimation",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    scenario_seed_owners={"sdr_categorical_k5_crossfit": "sdr_categorical_k5"},
    scenario_n=SCENARIO_N,
    resampling_seed=RESAMPLING_SEED,
    implementation="cleverly-long-horizon-sdr",
    pending_reference="lmtp",
    pending_reason=(
        "X59: R lmtp comparator runs deferred by the owner on 2026-10-14; X39 accepted on Python property evidence"
    ),
    modules=(
        "tests/studies/canonical_long_horizon_sdr.py",
        "tests/studies/long_horizon_sdr_properties.py",
        "tests/studies/long_horizon_primary.py",
        "tests/studies/long_horizon_common.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal_horizon.py",
        "tests/longitudinal_horizon_plans.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_long_horizon_sdr",
    properties_module="tests.studies.long_horizon_sdr_properties",
    publication_policy="reporting",
)

CONFIGURATION = {
    "estimator": ESTIMATOR,
    "n_folds": {scenario.name: scenario.n_folds for scenario in SCENARIOS},
    "g_bounds": list(G_BOUNDS),
    "outcome_learner": "QuasiBinomialGLM",
    "pseudo_learner": "LinearRegression (unclipped)",
    "mechanism": "the law's, passed as probability columns",
    "laws": {
        scenario.name: [scenario.law.B, scenario.law.s, scenario.law.c0] for scenario in SCENARIOS
    },
    "plans": {scenario.name: list(scenario.plans) for scenario in SCENARIOS},
}


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """The primary phase: every scenario at its declared size, or every one at an override ``n``."""
    return run_primary(STUDY, SCENARIOS, ESTIMATOR, replicates=replicates, n=n, n_jobs=n_jobs)
