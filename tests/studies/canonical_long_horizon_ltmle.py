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
``binary_k5_msm``    L_R, n = 96,000: ``dur0`` to ``dur5``, the working model intercept +
                     duration (identity link, unit weights); its two coefficients only
===================  ===========================================================================

Each scenario runs at the largest exact-law tail size :math:`n_\kappa` of the rows it gates,
rounded up to 4,000 (``tests/unit/test_long_horizon_design.py``).  The MSM scenario runs at twice
its largest :math:`n_\kappa` (94,226 for the intercept), rounded up to 96,000, so that both
coefficients are in the Wald regime.  The mechanism is the law's,
passed as probability columns; the outcome regressions are the declared quasibinomial GLM.

**Comparator.**  R ``ltmle`` 1.3-0 is the declared comparator, and its runs are pending: the record
declares ``pending_reference="ltmle"`` and keeps ``reference=None``.  The primary phase writes no
sample.  ``tests/canonical/long_horizon_ltmle/run_study.R`` is a stub that names the owning item.

**Intervals** (plan rev 4.4).  A single-fold, non-MSM scenario fits with ``n_bootstrap=199`` and
``bootstrap_interval="bca"``.  It writes a Wald row and, under the estimand name plus ``__bca``,
a BCa row whose standard error is the bootstrap SD.  A Wald row gates only where the scenario n
is at least twice the row's :math:`n_\kappa` (decision C1, section 5.1).  A BCa row gates only
where the Wald row does not (n below twice the row's :math:`n_\kappa`, where the Wald interval
fails), and reports where it does.  The cross-fitted scenario reports Wald rows only, because a
cross-fitted bootstrap is experimental (X20).  The MSM scenario reports Wald coefficients only
and gates both, because its n is at least twice their tail sizes.  Each ``interval_calibration`` cell runs
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
the simultaneous band                  The Wald-curve (Rademacher multiplier) band of the
(``simultaneous_coverage``)            influence curves (cell ``k5__simultaneous_band``).  Its
                                       rev 4.2 pilot passed (joint coverage 0.9425, R = 4,000).
                                       Its 99 percent joint-coverage interval leaves (0.92,
                                       0.98).  Defect suspect on either side.  The pointwise
                                       control must stay below nominal.  The study fits no
                                       bootstrap band.
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
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.long_horizon_common import G_BOUNDS
from tests.studies.long_horizon_primary import Scenario, declared, run_primary
from tests.studies.long_horizon_primary import draw_from_seed as _draw_from_seed

#: The truth-row rule of the design picks R from 1,200, 2,000 and 3,000 (plans
#: ``X39-pilot-merge-reading.md``, ``_x39/impl/truth_pooled.log``).
PRIMARY_REPLICATES = 3_000
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
        96_000,
        tuple(f"dur{j}" for j in range(6)),
        msm=True,
    ),
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
    "censoring_necessity": (
        "policy__k5__declared",
        "policy__k5__ignored",
    ),
    "double_robustness": (
        "respond_vs_ref__k5__both_correct",
        "respond_vs_ref__k5__both_wrong",
        "respond_vs_ref__k5__mechanism_correct",
        "respond_vs_ref__k5__outcome_correct",
    ),
    "interval_calibration": (
        "msm_duration__k5__known_mechanism",
        "msm_duration__k5__noise_control",
        "msm_duration__k5__shrunken_se_control",
        "policy__k3__known_mechanism",
        "policy__k3__noise_control",
        "policy__k3__shrunken_se_control",
        "policy__k4__known_mechanism",
        "policy__k4__noise_control",
        "policy__k4__shrunken_se_control",
        "policy__k5__known_mechanism",
        "policy__k5__noise_control",
        "policy__k5__shrunken_se_control",
        "policy_vs_policylast__k3__inflated_se_control",
        "policy_vs_policylast__k3__known_mechanism",
        "policy_vs_policylast__k3__noise_control",
        "policy_vs_policylast__k3__shrunken_se_control",
        "policy_vs_policylast__k4__inflated_se_control",
        "policy_vs_policylast__k4__known_mechanism",
        "policy_vs_policylast__k4__noise_control",
        "policy_vs_policylast__k4__shrunken_se_control",
        "policy_vs_policylast__k5__inflated_se_control",
        "policy_vs_policylast__k5__known_mechanism",
        "policy_vs_policylast__k5__noise_control",
        "policy_vs_policylast__k5__shrunken_se_control",
        "policy_vs_ref__k3__known_mechanism",
        "policy_vs_ref__k3__noise_control",
        "policy_vs_ref__k3__shrunken_se_control",
        "policy_vs_ref__k4__known_mechanism",
        "policy_vs_ref__k4__noise_control",
        "policy_vs_ref__k4__shrunken_se_control",
        "policy_vs_ref__k5__known_mechanism",
        "policy_vs_ref__k5__noise_control",
        "policy_vs_ref__k5__shrunken_se_control",
        "respond_vs_ref__k3__known_mechanism",
        "respond_vs_ref__k3__noise_control",
        "respond_vs_ref__k3__shrunken_se_control",
        "respond_vs_ref__k4__known_mechanism",
        "respond_vs_ref__k4__noise_control",
        "respond_vs_ref__k4__shrunken_se_control",
        "respond_vs_ref__k5__known_mechanism",
        "respond_vs_ref__k5__noise_control",
        "respond_vs_ref__k5__shrunken_se_control",
        "static_contrast__k3__known_mechanism",
        "static_contrast__k3__noise_control",
        "static_contrast__k3__shrunken_se_control",
        "static_contrast__k4__known_mechanism",
        "static_contrast__k4__noise_control",
        "static_contrast__k4__shrunken_se_control",
        "static_contrast__k5__known_mechanism",
        "static_contrast__k5__noise_control",
        "static_contrast__k5__shrunken_se_control",
    ),
    "learned_mechanism": ("respond_vs_ref__k5__learned",),
    "policy_necessity": (
        "policy__k5__declared_policy",
        "policy__k5__mix_reversed_control",
    ),
    "power": ("policy_vs_policylast__k5__alternative",),
    "root_n_and_efficiency": (
        "static_contrast__k4__n_12000",
        "static_contrast__k4__n_4000",
        "static_contrast__k4__n_48000",
    ),
    "root_n_rate": (
        "static_contrast__k4__empirical_sd",
        "static_contrast__k4__reported_se",
    ),
    "rule_necessity": (
        "respond__k5__declared_rule",
        "respond__k5__reversed_rule",
    ),
    "simultaneous_coverage": (
        "k5__pointwise_joint_control",
        "k5__simultaneous_band",
    ),
    "support_loss": (
        "policy__k5__n4000__default_bounds",
        "policy__k5__n4000__no_bounds",
        "static__k5__n4000__default_bounds",
        "static__k5__n4000__no_bounds",
    ),
    "targeting_necessity": (
        "policy__k3__targeted",
        "policy__k3__untargeted",
        "policy__k4__targeted",
        "policy__k4__untargeted",
        "policy__k5__targeted",
        "policy__k5__untargeted",
        "respond__k3__targeted",
        "respond__k3__untargeted",
        "respond__k4__targeted",
        "respond__k4__untargeted",
        "respond__k5__targeted",
        "respond__k5__untargeted",
    ),
    "type_i_error": ("policy_vs_policylast__k5__sharp_null",),
}

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
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    bias_row_families=BIAS_ROW_FAMILIES,
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


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``scenario`` from ``seed``, with the exact truth of every estimand."""
    return _draw_from_seed(SCENARIOS, scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's own seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)
