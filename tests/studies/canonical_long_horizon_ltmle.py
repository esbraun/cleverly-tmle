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

**Declared red-cell reading** (plan section 5.6, before any run).  The study publishes under
``reporting``.  Its gated truth rows (``GATED_ROWS``) and gated property cells (``GATED_CELLS``)
are listed in :mod:`tests.long_horizon_design`.  ``tests/unit/test_long_horizon_design.py`` fails
on any red member of either list once the study commits its summaries.  A row applies to the
study that declares its family.

=====================================  =========================================================
red cell                               reading
=====================================  =========================================================
a primary truth row                    Only a gated row blocks: a row gates in its gating
                                       scenario alone, and the same quantity in another
                                       scenario reports.  Diagnose.  A defect is fixed and the
                                       run repeats under a fresh declaration.
a calibration positive                 Finite-sample only when its 99 percent SE-ratio interval
(``interval_calibration``)             contains 1, its coverage interval contains 0.95, and the
                                       diagnosis finds no defect; it then goes to an owner row.
                                       Any other red calibration row is a defect suspect:
                                       diagnose, fix, re-run under a fresh declaration.  An
                                       unfixed defect never ends as an owner row.
a bias-family positive                 Its 99 percent standardized-bias interval does not lie
(the necessity families,               inside the margin.  Defect suspect: diagnose the
``sequential_double_robustness``,      estimator, then the law and the learner rung of the
``double_robustness``)                 design test.  Finite-sample only when the diagnosis finds
                                       no defect and the interval is wider than the margin
                                       through R alone; it then goes to an owner row.  The run
                                       never adds replicates to a red positive.
a control that passes                  A design defect, because its exact displacement is at
                                       least 3 sampling SDs.  Diagnose the estimator and the
                                       design test first.  Fix the control or the law under a
                                       fresh declaration, never by buying replications or
                                       lowering its floor.
the simultaneous band                  Its 99 percent joint-coverage interval leaves (0.92,
(``simultaneous_coverage``)            0.98).  Defect suspect on either side: below, the critical
                                       value or the tail regime is wrong; above, the band is
                                       conservative and the diagnosis names why.  The pointwise
                                       control must stay below nominal.
the overfitting positive               Its SE-ratio interval leaves [0.85, 1.20], or the coverage
(``crossfit_overfitting``)             gain over the in-sample control falls short.  Defect
                                       suspect: own-row reuse across the K nodes.  The in-sample
                                       control must fall below the SE-ratio ceiling.
a root-n rung or slope row             A positive rung (12,000 and 48,000) is red when its
(``root_n_and_efficiency``,            coverage lower limit is below 0.90, its SE-ratio interval
``root_n_rate``)                       leaves (0.80, 1.20), or its bias is not established
                                       inside the margin.  A slope row is red when its interval
                                       is not within 0.125 of -0.5 or reaches -0.25.  Defect
                                       suspect, with the rung pilots read first.  The 4,000 rung
                                       carries no claim at R = 1,000, because its Clopper-
                                       Pearson interval is narrower than the 0.05 gap between
                                       the coverage floor and nominal; the page does not list it
                                       as a control.
type I error and power                 Size is red when the upper limit of its rejection
(``type_i_error``, ``power``)          interval exceeds 0.10 or its coverage lower limit falls
                                       below 0.90.  Power is red when the lower limit of its
                                       rejection rate is below 0.80.  Defect suspect; the sharp
                                       null has truth 0 exactly.
a ``support_loss`` or                  Reporting; no verdict.  The page states the measured
``learned_mechanism`` cell             values, with the measured binding share per node, beside
                                       the exact-law prediction.  The two-sided tripwire of the
                                       plan makes a cell a defect suspect.
a recorded fitting failure             Accounted by RM39b2's ``require_accounted`` under
                                       ``FAILURE_BOUND`` = 0.01 and the module's declared clause
                                       tree.  A cell over the bound is red and diagnosed.
=====================================  =========================================================

No budget, margin, law, learner, fold count or seed moves after a verdict is seen, except under
the diagnosed-defect route with a fresh declaration commit.
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
