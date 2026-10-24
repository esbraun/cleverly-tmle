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
``sdr_policy_k5``                   L_R, n = 44,000: ``low``, ``mix``, ``taper``
``sdr_policy_k5_mix``               (1, 0.05, 3.5), n = 96,000: ``low``, ``mix``, ``taper``
``sdr_policy_k5_taper``             (1, 0, 3.5), n = 76,000: ``low``, ``taper``
==================================  ============================================================

The policy scenarios also report ``taper`` against ``mix``, and the categorical ones ``mix``
against ``mixlast``.

**Comparator.**  R ``lmtp`` is the declared comparator, and its runs are pending: the record
declares ``pending_reference="lmtp"`` and keeps ``reference=None``.  The primary phase writes no
sample.  ``tests/canonical/long_horizon_sdr/run_study.R`` is a stub that names the owning item.

**Intervals** (plan rev 4.4).  A single-fold, non-MSM scenario fits with ``n_bootstrap=199`` and
``bootstrap_interval="bca"``.  It writes a Wald row and, under the estimand name plus ``__bca``,
a BCa row whose standard error is the bootstrap SD.  A Wald row gates only where the scenario n
is at least twice the row's :math:`n_\kappa` (decision C1, section 5.1).  A BCa row gates only
where the Wald row does not (n below twice the row's :math:`n_\kappa`), and reports where it does.
The cross-fitted scenario reports Wald rows only, because a cross-fitted bootstrap is experimental (X20).  Each ``interval_calibration`` cell runs
at ``n = round_n(2 n_kappa)`` on its law.

**Declared red-cell reading** (plan section 5.6, before any run).  The study publishes under
``reporting``.  Its gated truth rows (``GATED_ROWS``) and gated property cells (``GATED_CELLS``)
are listed in :mod:`tests.long_horizon_design`.  ``tests/unit/test_long_horizon_design.py`` fails
on any red member of either list once the study commits its summaries.  A row applies to the
study that declares its family.

=====================================  =========================================================
red cell                               reading
=====================================  =========================================================
a primary truth row                    Only a gated row blocks: a row gates in its gating
(Wald or ``__bca``)                    scenario alone, and the same quantity in another
                                       scenario reports.  A Wald row gates only at n >= 2
                                       n_kappa.  Diagnose.  A defect is fixed and the run
                                       repeats under a fresh declaration.  A Wald calibration
                                       cell that fails at 2 n_kappa is a red cell: the factor 2
                                       is not re-tuned.
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
the simultaneous band                  This study has no band cell.  Its bootstrap band stays
(``simultaneous_coverage``)            experimental and gates no row.
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

import pandas as pd

from tests import discrete_law_longitudinal_horizon as law_
from tests.discrete_law_longitudinal_horizon import Law
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.long_horizon_common import G_BOUNDS
from tests.studies.long_horizon_primary import Scenario, declared, run_primary
from tests.studies.long_horizon_primary import draw_from_seed as _draw_from_seed

#: The truth-row rule of the design picks R from 1,200, 2,000 and 3,000 (plans
#: ``X39-pilot-merge-reading.md``, ``_x39/impl/truth_pooled.log``).
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
    Scenario("sdr_policy_k5", 5, 3, law_.L_R, 44_000, POLICY, paired=TAPER),
    Scenario("sdr_policy_k5_mix", 5, 3, Law(1.0, 0.05, 3.5), 96_000, POLICY, paired=TAPER),
    Scenario("sdr_policy_k5_taper", 5, 3, Law(1.0, 0.0, 3.5), 76_000, ("low", "taper")),
)
SCENARIO_ESTIMANDS, SCENARIO_N = declared(SCENARIOS)
PRIMARY_N = min(SCENARIO_N.values())

#: Property family -> the cells the committed property summary must contain.
#: Every bias family of the long-horizon studies reads its rows one at a time.
BIAS_ROW_FAMILIES = frozenset(
    {
        "targeting_necessity",
        "censoring_necessity",
        "categorical_probability_necessity",
        "rule_necessity",
        "policy_necessity",
        "sequential_double_robustness",
    }
)

PROPERTY_CELLS = {
    "categorical_probability_necessity": (
        "third_arm__k5__assigned_probability",
        "third_arm__k5__binary_complement",
    ),
    "censoring_necessity": (
        "binary_policy__k5__declared",
        "binary_policy__k5__ignored",
        "categorical_policy__k5__declared",
        "categorical_policy__k5__ignored",
    ),
    "crossfit_overfitting": (
        "policy_vs_ref__k5__cross_fitted",
        "policy_vs_ref__k5__in_sample_control",
    ),
    "interval_calibration": (
        "binary_policy__k3__known_mechanism",
        "binary_policy__k3__noise_control",
        "binary_policy__k3__shrunken_se_control",
        "binary_policy__k4__known_mechanism",
        "binary_policy__k4__noise_control",
        "binary_policy__k4__shrunken_se_control",
        "binary_policy__k5__known_mechanism",
        "binary_policy__k5__noise_control",
        "binary_policy__k5__shrunken_se_control",
        "binary_policy_vs_policylast__k3__inflated_se_control",
        "binary_policy_vs_policylast__k3__known_mechanism",
        "binary_policy_vs_policylast__k3__noise_control",
        "binary_policy_vs_policylast__k3__shrunken_se_control",
        "binary_policy_vs_policylast__k4__inflated_se_control",
        "binary_policy_vs_policylast__k4__known_mechanism",
        "binary_policy_vs_policylast__k4__noise_control",
        "binary_policy_vs_policylast__k4__shrunken_se_control",
        "binary_policy_vs_policylast__k5__inflated_se_control",
        "binary_policy_vs_policylast__k5__known_mechanism",
        "binary_policy_vs_policylast__k5__noise_control",
        "binary_policy_vs_policylast__k5__shrunken_se_control",
        "binary_policy_vs_ref__k3__known_mechanism",
        "binary_policy_vs_ref__k3__noise_control",
        "binary_policy_vs_ref__k3__shrunken_se_control",
        "binary_policy_vs_ref__k4__known_mechanism",
        "binary_policy_vs_ref__k4__noise_control",
        "binary_policy_vs_ref__k4__shrunken_se_control",
        "binary_policy_vs_ref__k5__known_mechanism",
        "binary_policy_vs_ref__k5__noise_control",
        "binary_policy_vs_ref__k5__shrunken_se_control",
        "binary_respond_vs_ref__k3__known_mechanism",
        "binary_respond_vs_ref__k3__noise_control",
        "binary_respond_vs_ref__k3__shrunken_se_control",
        "binary_respond_vs_ref__k4__known_mechanism",
        "binary_respond_vs_ref__k4__noise_control",
        "binary_respond_vs_ref__k4__shrunken_se_control",
        "binary_respond_vs_ref__k5__known_mechanism",
        "binary_respond_vs_ref__k5__noise_control",
        "binary_respond_vs_ref__k5__shrunken_se_control",
        "binary_static_contrast__k3__known_mechanism",
        "binary_static_contrast__k3__noise_control",
        "binary_static_contrast__k3__shrunken_se_control",
        "binary_static_contrast__k4__known_mechanism",
        "binary_static_contrast__k4__noise_control",
        "binary_static_contrast__k4__shrunken_se_control",
        "binary_static_contrast__k5__known_mechanism",
        "binary_static_contrast__k5__noise_control",
        "binary_static_contrast__k5__shrunken_se_control",
        "categorical_policy__k3__known_mechanism",
        "categorical_policy__k3__noise_control",
        "categorical_policy__k3__shrunken_se_control",
        "categorical_policy__k4__known_mechanism",
        "categorical_policy__k4__noise_control",
        "categorical_policy__k4__shrunken_se_control",
        "categorical_policy__k5__known_mechanism",
        "categorical_policy__k5__noise_control",
        "categorical_policy__k5__shrunken_se_control",
        "categorical_policy_vs_policylast__k3__inflated_se_control",
        "categorical_policy_vs_policylast__k3__known_mechanism",
        "categorical_policy_vs_policylast__k3__noise_control",
        "categorical_policy_vs_policylast__k3__shrunken_se_control",
        "categorical_policy_vs_policylast__k4__inflated_se_control",
        "categorical_policy_vs_policylast__k4__known_mechanism",
        "categorical_policy_vs_policylast__k4__noise_control",
        "categorical_policy_vs_policylast__k4__shrunken_se_control",
        "categorical_policy_vs_policylast__k5__inflated_se_control",
        "categorical_policy_vs_policylast__k5__known_mechanism",
        "categorical_policy_vs_policylast__k5__noise_control",
        "categorical_policy_vs_policylast__k5__shrunken_se_control",
        "categorical_policy_vs_policylast__k5_crossfit__inflated_se_control",
        "categorical_policy_vs_policylast__k5_crossfit__known_mechanism",
        "categorical_policy_vs_policylast__k5_crossfit__noise_control",
        "categorical_policy_vs_policylast__k5_crossfit__shrunken_se_control",
        "categorical_policy_vs_ref__k3__known_mechanism",
        "categorical_policy_vs_ref__k3__noise_control",
        "categorical_policy_vs_ref__k3__shrunken_se_control",
        "categorical_policy_vs_ref__k4__known_mechanism",
        "categorical_policy_vs_ref__k4__noise_control",
        "categorical_policy_vs_ref__k4__shrunken_se_control",
        "categorical_policy_vs_ref__k5__known_mechanism",
        "categorical_policy_vs_ref__k5__noise_control",
        "categorical_policy_vs_ref__k5__shrunken_se_control",
        "categorical_policy_vs_ref__k5_crossfit__known_mechanism",
        "categorical_policy_vs_ref__k5_crossfit__noise_control",
        "categorical_policy_vs_ref__k5_crossfit__shrunken_se_control",
        "categorical_respond_vs_ref__k3__known_mechanism",
        "categorical_respond_vs_ref__k3__noise_control",
        "categorical_respond_vs_ref__k3__shrunken_se_control",
        "categorical_respond_vs_ref__k4__known_mechanism",
        "categorical_respond_vs_ref__k4__noise_control",
        "categorical_respond_vs_ref__k4__shrunken_se_control",
        "categorical_respond_vs_ref__k5__known_mechanism",
        "categorical_respond_vs_ref__k5__noise_control",
        "categorical_respond_vs_ref__k5__shrunken_se_control",
        "categorical_static_contrast__k3__known_mechanism",
        "categorical_static_contrast__k3__noise_control",
        "categorical_static_contrast__k3__shrunken_se_control",
        "categorical_static_contrast__k4__known_mechanism",
        "categorical_static_contrast__k4__noise_control",
        "categorical_static_contrast__k4__shrunken_se_control",
        "categorical_static_contrast__k5__known_mechanism",
        "categorical_static_contrast__k5__noise_control",
        "categorical_static_contrast__k5__shrunken_se_control",
    ),
    "learned_mechanism": (
        "binary_respond_vs_ref__k5__learned",
        "categorical_respond_vs_ref__k5__learned",
    ),
    "policy_necessity": (
        "binary_policy__k5__declared_policy",
        "binary_policy__k5__mix_reversed_control",
        "categorical_policy__k5__declared_policy",
        "categorical_policy__k5__mix_node1_control",
    ),
    "root_n_and_efficiency": (
        "respond_vs_ref__k4__n_12000",
        "respond_vs_ref__k4__n_4000",
        "respond_vs_ref__k4__n_48000",
    ),
    "root_n_rate": (
        "respond_vs_ref__k4__empirical_sd",
        "respond_vs_ref__k4__reported_se",
    ),
    "sequential_double_robustness": (
        "P_even",
        "P_even__n4000",
        "P_odd",
        "P_odd__n4000",
        "ltmle_P_odd_control",
        "odd_nodes_both_wrong_control",
    ),
    "support_loss": (
        "binary_policy__k5__n4000__default_bounds",
        "binary_policy__k5__n4000__no_bounds",
        "binary_static__k5__n4000__default_bounds",
        "binary_static__k5__n4000__no_bounds",
        "categorical_policy__k5__n4000__default_bounds",
        "categorical_policy__k5__n4000__no_bounds",
        "categorical_static__k5__n16000__no_bounds",
        "categorical_static__k5__n4000__default_bounds",
        "categorical_static__k5__n4000__no_bounds",
        "categorical_static__k5__n64000__no_bounds",
    ),
}

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
    # No comparison_overrides while the runs are pending. X59 declares
    # "sdr_categorical_k5_crossfit:*": "different estimator" with the runs.
    comparison="matching implementation",
    modules=(
        "tests/studies/canonical_long_horizon_sdr.py",
        "tests/studies/long_horizon_sdr_properties.py",
        "tests/studies/long_horizon_properties.py",
        "tests/studies/canonical_long_horizon_ltmle.py",
        "tests/studies/canonical_long_horizon_categorical_ltmle.py",
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
    runner_module="tests.studies.canonical_long_horizon_sdr",
    properties_module="tests.studies.long_horizon_sdr_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    bias_row_families=BIAS_ROW_FAMILIES,
    publication_policy="reporting",
    failure_bound={"cleverly-long-horizon-sdr": 0.01, "lmtp": 0.01},
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


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``scenario`` from ``seed``, with the exact truth of every estimand."""
    return _draw_from_seed(SCENARIOS, scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's own seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)
