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

**Comparator** (X59).  R ``lmtp`` 9996b04 is the reference, run reference-only on the first 100
replicates of each paired scenario (``tests/canonical/long_horizon_categorical_ltmle/run_study.R``).
Its exact claim is the G1 gate on ``mix`` (``parity.csv``); every margins pair is descriptive.

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
the simultaneous band                  The studentized bootstrap sup-t band at B = 199 (cell
(``simultaneous_coverage``)            ``k5__simultaneous_band``).  Its 99 percent joint-
                                       coverage interval leaves (0.92, 0.98).  Defect suspect
                                       on either side: below, the critical value or the tail
                                       regime is wrong; above, the band is conservative and
                                       the diagnosis names why.  A stop means diagnose the
                                       band, not switch its construction.  The pointwise
                                       control must stay below nominal.
the Wald-curve band                    Reporting; no verdict.  Cell ``k5__ic_band`` publishes
(``k5__ic_band``)                      the multiplier band of the influence curves beside the
                                       gated band.
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
a red G1 cell                          Blocks registration (plan section 4.3): the 1e-6
(``parity.csv``)                       tolerance, the prefix and the seeds do not move.
                                       Diagnose the subject, the adapter, the shared input and
                                       ``lmtp`` 9996b04; fix a defect and re-run under a fresh
                                       declaration.  G1 covers ``mix`` in the four one-fold
                                       scenarios that fit it (smoke maximum 8.1e-11).
a scenario the 24-hour cap cut         Each scenario is charged before it starts at the sizing
(``--reference-deadline``)             rule's cost model and pairs the longest prefix predicted
                                       to end by the deadline; the run's manifest gives each cut
                                       a limit line.  At 50 or more kept replicates
                                       (``EXACT_FLOOR``) its exact cells are graded at the kept
                                       prefix.  Under 50 they report their differences without a
                                       verdict.  With no prefix of 2 the scenario is unpaired
                                       and the page states the gap.  The registration declares
                                       the kept counts; no cut scenario is re-run to its
                                       declared prefix after a verdict.
a recorded comparator failure          Accounted under ``FAILURE_BOUND`` = 0.01 per cell: at the
(the run's ``failures.csv``)           declared prefix of 100 one failed replicate is allowed,
                                       and a second makes every cell of its scenario red; a cut
                                       prefix under 100 allows none.  A refusal on either side
                                       counts, never a skip.
a fluctuation guard record             The adapter stops a replicate whose lmtp fluctuation
(``lmtp_9996b04_crossfit_adapter.R``)  reports a non-finite epsilon or one above 1e3, so a
                                       diverged fit is a recorded failure, never a published
                                       row.  It counts under ``FAILURE_BOUND``.  The main run
                                       counts guard records per scenario after this study
                                       (``x59_smoke guard``) and stops above 1 per 100 for
                                       diagnosis.
a margins pair                         Published with its four legs and no verdict.  No pass
(``paired_role="descriptive"``)        probability is computed for these legs at the declared
                                       prefix, so the page does not say in advance which legs
                                       were resolvable; it reports each leg's interval as
                                       measured.
an unpaired cell                       Six plans ``lmtp`` 9996b04's node-K fluctuation cannot
                                       fit (``UNPAIRED_PLANS``): ``categorical_k4`` ``switch``,
                                       ``categorical_k4_high`` ``high``,
                                       ``categorical_k5_static`` ``high`` and ``switch``,
                                       ``categorical_k5_switch`` ``switch`` and
                                       ``categorical_k5_high`` ``high``, with their contrasts.
                                       ``categorical_k5_crossfit`` has no counterpart estimator.
                                       The page states each gap as a comparator limitation.
a cell not promoted to G1              Every paired cell but ``mix`` stays descriptive: the
                                       smoke paired 19 of 38 cells above 1e-10 (largest 2.0e-9),
                                       so the rev 2.1 promotion rule does not reach them.
=====================================  =========================================================

No budget, margin, law, learner, fold count or seed moves after a verdict is seen, except under
the diagnosed-defect route with a fresh declaration commit.
"""

from __future__ import annotations

import pandas as pd

from tests import discrete_law_longitudinal_horizon as law_
from tests.discrete_law_longitudinal_horizon import Law
from tests.parallel import STUDY_JOBS
from tests.studies import long_horizon_reference as reference_
from tests.studies.evidence import replication
from tests.studies.evidence.parity import ParityCell, ParityDeclaration, ReferenceSample
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.long_horizon_common import G_BOUNDS
from tests.studies.long_horizon_primary import Scenario, declared, run_primary
from tests.studies.long_horizon_primary import draw_from_seed as _draw_from_seed

#: The truth-row rule of the design picks R from 1,200, 2,000 and 3,000 (plans
#: ``X39-pilot-merge-reading.md``, ``_x39/impl/truth_pooled.log``).
PRIMARY_REPLICATES = 2_000
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

#: Property family -> the cells the committed property summary must contain.
#: Every bias family of the long-horizon studies reads its rows one at a time.
BIAS_ROW_FAMILIES = frozenset(
    {
        "targeting_necessity",
        "censoring_necessity",
        "categorical_probability_necessity",
        "rule_necessity",
        "policy_necessity",
    }
)

PROPERTY_CELLS = {
    "categorical_probability_necessity": (
        "third_arm__k5__assigned_probability",
        "third_arm__k5__binary_complement",
    ),
    "censoring_necessity": (
        "policy__k5__declared",
        "policy__k5__ignored",
    ),
    "crossfit_overfitting": (
        "policy_vs_ref__k5__cross_fitted",
        "policy_vs_ref__k5__in_sample_control",
    ),
    "double_robustness": (
        "respond_vs_ref__k5__both_correct",
        "respond_vs_ref__k5__both_wrong",
        "respond_vs_ref__k5__mechanism_correct",
        "respond_vs_ref__k5__outcome_correct",
    ),
    "interval_calibration": (
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
        "policy_vs_policylast__k5_crossfit__inflated_se_control",
        "policy_vs_policylast__k5_crossfit__known_mechanism",
        "policy_vs_policylast__k5_crossfit__noise_control",
        "policy_vs_policylast__k5_crossfit__shrunken_se_control",
        "policy_vs_ref__k3__known_mechanism",
        "policy_vs_ref__k3__noise_control",
        "policy_vs_ref__k3__shrunken_se_control",
        "policy_vs_ref__k4__known_mechanism",
        "policy_vs_ref__k4__noise_control",
        "policy_vs_ref__k4__shrunken_se_control",
        "policy_vs_ref__k5__known_mechanism",
        "policy_vs_ref__k5__noise_control",
        "policy_vs_ref__k5__shrunken_se_control",
        "policy_vs_ref__k5_crossfit__known_mechanism",
        "policy_vs_ref__k5_crossfit__noise_control",
        "policy_vs_ref__k5_crossfit__shrunken_se_control",
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
        "policy__k5__mix_node1_control",
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
    "rule_necessity": (
        "respond__k3__declared_rule",
        "respond__k3__reversed_rule",
    ),
    "simultaneous_coverage": (
        "k5__ic_band",
        "k5__pointwise_joint_control",
        "k5__simultaneous_band",
    ),
    "support_loss": (
        "policy__k5__n4000__default_bounds",
        "policy__k5__n4000__no_bounds",
        "static__k5__n16000__no_bounds",
        "static__k5__n4000__default_bounds",
        "static__k5__n4000__no_bounds",
        "static__k5__n64000__no_bounds",
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
}

#: The plans ``lmtp`` ``9996b04`` cannot fit on this study's inputs.  Its node-K fluctuation
#: (``fluc`` in R/tmle.R) is ``glm(y ~ offset(qlogis(Q)), weights = cumulative ratios, family =
#: binomial)`` at R's default start; under node-K weights of about 1e3 glm.fit overshoots from
#: that start, reports convergence at an epsilon of about 3e15, and every targeted prediction
#: and every earlier pseudo-outcome becomes 0.99999.  The same fit on the weights rescaled to
#: mean one, whose maximum-likelihood estimate is the same, converges (2.014 on
#: ``categorical_k5_static`` ``high``, replicate 0), and ``cleverly``'s fit of these cells
#: converges.  A comparator limitation, not a ``cleverly`` result.  Evidence:
#: ``tests/canonical/lmtp_9996b04/fluctuation_scan.py`` on replicates 0 to 99, a default-start
#: fit against the rescaled one, diverged when they differ by more than 1e-6, node K only:
#: ``categorical_k5_static`` ``high`` 100/100 and ``switch`` 100/100, ``categorical_k5_switch``
#: ``switch`` 100/100, ``categorical_k5_high`` ``high`` 100/100, ``categorical_k4`` ``switch``
#: 70/100 and ``categorical_k4_high`` ``high`` 10/100.  Every other one-fold plan, every G1
#: ``mix`` cell among them, diverged on 0/100.  All six are descriptive plans, and the decision
#: was taken on the smoke inputs before any parity output.  That scan is an intercept-only
#: proxy at node K.  The exact check ran for the declaration (pre-queue review item 7):
#: ``fluctuation_emulation_scan.py`` runs lmtp 9996b04's whole one-fold TMLE loop
#: (``fluctuation_emulation.R``, host R 4.3.3) on the full input of replicates 0 to 99 of every
#: paired plan here and 0 to 499 of ``long-horizon-lmtp-parity`` ``parity_tmle_mix``.  No node
#: of any of the 2,700 plan-replicates diverged (largest ``|epsilon|`` 0.45) and no targeted
#: follower cell left ``Q_BOUNDS`` (``lmtp_9996b04/emulation-summary.csv``).  The paired
#: adapters also record a replicate as failed when any node's epsilon is non-finite or above
#: 1e3 (lmtp's ``fluc`` guard, ``lmtp_9996b04_crossfit_adapter.R``), for whatever the main run
#: meets beyond these replicates.
UNPAIRED_PLANS = {
    "categorical_k4": ("switch",),
    "categorical_k4_high": ("high",),
    "categorical_k5_static": ("high", "switch"),
    "categorical_k5_switch": ("switch",),
    "categorical_k5_high": ("high",),
}
#: The comparator's cells (X59 plan section 2.2): every Wald row of every one-fold scenario but
#: those that read an unpaired plan.  The five-fold scenario has no counterpart: ``cleverly``
#: fits the pooled update and ``lmtp`` ``9996b04`` a per-fold one (section 4.4), so it stays
#: unpaired.  The runner fits only the plans its truth rows name, so it never fits these.
REFERENCE_SCENARIOS = reference_.reported_cells(
    SCENARIOS, unpaired_plans=UNPAIRED_PLANS, unpaired_scenarios=("categorical_k5_crossfit",)
)
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
    reference="lmtp",
    # X59: the comparator pairs the cells above on the first 100 replicates of each paired
    # scenario, the section 6 sizing rule's output (tests/canonical/x59_sizing).  Every
    # margins pair is descriptive; the exact claims are the parity gates (``PARITY``).
    reference_scenarios=REFERENCE_SCENARIOS,
    reference_replicates=dict.fromkeys(REFERENCE_SCENARIOS, 100),
    paired_role="descriptive",
    accepted_reference_failure=reference_.PREFIX_REFERENCE_FAILURE,
    comparison="matching implementation",
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
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    bias_row_families=BIAS_ROW_FAMILIES,
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


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``scenario`` from ``seed``, with the exact truth of every estimand."""
    return _draw_from_seed(SCENARIOS, scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's own seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ----------------------------------------------------------------------------- the X59 comparator
#: The comparator's declared environment: no mutation, and the declared learner (M2).
REFERENCE_ENVIRONMENT = reference_.LMTP_ENVIRONMENT
#: The comparator's pins, for the manifest's ``generated_with.reference`` block.
REFERENCE_METADATA = reference_.LMTP_METADATA
#: The section 6 priority tier of each paired scenario: 2 holds exact cells, 4 margins only.
#: A tier-2 scenario's descriptive plans are fitted, and charged, at the tier-2 prefix.
REFERENCE_TIERS = {
    name: 2 if "ey_regimen[mix]" in cells else 4 for name, cells in REFERENCE_SCENARIOS.items()
}
#: G1 on ``ey_regimen[mix]``, the all-policy plan, in every one-fold scenario that fits it.  The
#: rev 2.1 promotion does not reach this study's other plans: a cell is promoted only when its
#: identity holds at 1e-10 on the G0 frame and on the smoke set, and the smoke paired 19 of
#: this study's 38 cells above 1e-10 (largest ``|d estimate|`` 2.02e-9).  Two ``glm`` fits
#: of the stratified node regressions stop at different iterates there, so those cells keep
#: their descriptive margins rows.
PARITY = ParityDeclaration(
    cells=tuple(
        ParityCell("G1", name, "ey_regimen[mix]")
        for name, tier in REFERENCE_TIERS.items()
        if tier == 2
    )
)


def reference_sample(scenario: str, n: int, replicate: int) -> ReferenceSample:
    """The R ``lmtp`` input of one replicate, from this study's seed stream."""
    return reference_.sample(STUDY, SCENARIOS, "lmtp_tmle", scenario, n, replicate)
