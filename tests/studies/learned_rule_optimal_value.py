r"""Registered evidence for the value of the optimal rule (X11 (f)).

The subject is ``TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"], n_folds=10,
cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`` with RM30's interaction
logistic outcome learner, and every other setting at the constructor's default. It reports
``ey_optimal_rule[learned rule]``, whose target is the fixed parameter
:math:`\Psi_{d_0}(P_0) = E_0\bar Q_0(d_0(W), W)`, through the shipped fold-evaluated CV-TMLE of
the learned-rule value (van der Laan and Luedtke 2015, Section 7.2, last paragraph). The truth is
:math:`\Psi_{d_0}` by quadrature, one number per law, so the record reads one truth per cell.

Every cell reads the error against :math:`\Psi_{d_0}`. The source's licence is condition C3f
(the learner's limit rule is :math:`d_0`), C7 (no unit's blip is 0, JCI condition (5)) and C8
(the regret :math:`\Psi_{d_0} - \tilde\psi_{0n}` is :math:`o_P(n^{-1/2})`, by Lemma 1's margin
and rate conditions), with the treatment mechanism known. A law cell gates only when its
learner's population limit meets C3f, C7 and the margin reading of C8 (:func:`gated`, read from
:data:`LIMITS`, which ``test_learned_rule_optimal_value_design`` recomputes). Four laws and
settings:

``non_exceptional``, known ``g``, n = 2,000
    RM30's blip ``0.1 + W1``. The declared learner is correct, so :math:`d_1 = d_0`; no unit ties;
    the blip has a bounded density at 0, so the margin exponent is 1, and with a parametric
    :math:`\bar Q` the regret is :math:`O_P(1/n)`. The cell gates: the primary rows and the
    ``interval_calibration`` family (SE x 0.8 and noise controls) read it.
``wrong_limit``, known ``g``, n = 2,500
    The even blip ``3 (W1^2 - 0.5)``. The declared learner's limit rule treats nobody, while
    :math:`d_0` treats 29.3% of the population, so C3f fails: the fit's target converges to
    :math:`\Psi_{d_1}`, 0.0452 below :math:`\Psi_{d_0}`, 3.29 standard errors of :math:`d_1`'s
    curve at n = 2,500 (W-f2). Its coverage of :math:`\Psi_{d_0}` must fall below the floor
    (``optimal_target_control``), the must-fail control of the same instrument. Its n is the
    smallest multiple of 500 at which the gap is at least 3 SEs (2,085 needed); the primary and
    the other cells keep n = 2,000, because every property row carries its own n.
``non_exceptional``, estimated ``g``, n = 2,000
    The same law with the declared logistic treatment model fitted (plan decision E2, admitted
    experimental). Reported, not gated: no row is added for it.
``partially_exceptional``, known ``g``, n = 2,000
    The hinge blip ``(W1 - 0.2)_+``, zero on a set of mass 0.6, so C7 fails. The outcome learner
    adds ``A (W1 - 0.2)_+`` to the declared products, so the model is correct and C3f holds off
    the tie set. Reported, not gated: the contract makes no claim at an exceptional law.

Each row also carries the data-adaptive :math:`\tilde\psi_{0n}` of the fit's own fold rules
(refit, rule L4), the regret, and the record's ``rule_disagreement``; the summary publishes their
means, and the disagreement's range, with no verdict.

The X11 wave-2 plan (``X11-wave2-plan.md`` rev 2.1 in the plans repository, section 6), its
review (finding 6), the implementation notes and review ``reviews/x11/X11-I-impl-notes.md`` and
``reviews/x11/X11-I-impl-review.md`` (the nine study-design answers), and the study design record
``reviews/x11/X11-I-study-design.md`` declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_optimal_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

base = law.base

#: The primary replications: the known-g fit of ``non_exceptional``. The primary gate reads its
#: bias leg (the 99% interval of the mean error inside 0.25 SDs), coverage floor and SE sanity
#: band against :math:`\Psi_{d_0}`. PRIMARY_REPLICATES is the smallest multiple of 400 of at least
#: the study's floor of 1,000 whose predictive probability that the bias leg and the coverage
#: floor both pass is at least 0.95, read on the first 4,000-draw pilot and on the 8,000 draws
#: pooled with the probe, and the larger of the two (``test_learned_rule_optimal_value_design``).
#: The bias leg reads the regret: at n = 2,000 the pilots' mean regret is 0.094 estimate SDs, so
#: the 99% bias interval is expected to exclude zero (near [-0.17, -0.04]) and still pass the
#: 0.25 SD leg.
#:
#: What the regret does to each verdict against :math:`\Psi_{d_0}`, at n = 2,000 (design
#: review, finding 3). Bias: the mean regret, about -0.10 estimate SDs, is the bias the primary
#: publishes; against :math:`\tilde\psi_{0n}` the bias is about -0.01 SDs. Coverage: a bias of
#: 0.10 SD costs about 0.001 of coverage at an SE ratio of 1, so a gap from 0.95 comes mostly
#: from the SE ratio. SE ratio: the regret's own spread (SD about 0.0015) widens the error's SD
#: by about 0.9%, so the ratio against :math:`\Psi_{d_0}` sits about 0.9% below the ratio
#: against :math:`\tilde\psi_{0n}` (0.984 against 0.993 on the pooled pilots). Both effects are
#: of order :math:`n^{-1/2}` relative to the SE under C8 with a parametric outcome model. The
#: target control's miss is the limit gap itself, and the calibration controls inherit the
#: positive's regret. No verdict reads :math:`\tilde\psi_{0n}`; the summary publishes the
#: readings against it as descriptive columns.
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20269151
RESAMPLING_SEED = 20269152

#: The primary law, read with the known treatment mechanism.
PRIMARY_LAW = "non_exceptional"
SCENARIOS = {PRIMARY_LAW: (law.ESTIMAND,)}

#: The gate of a law cell (impl review, study-design answer 1), read at its learner's
#: population limit by quadrature: C7, no tie mass; C3f, the limit rule differs from
#: :math:`d_0` off the tie set on at most :data:`MAX_DISAGREEMENT` of the population (eight
#: grid-point masses of the quadrature, far below the control's 0.293); C8 with margin
#: exponent 1, the largest ``P0(0 < |b0| <= t) / t`` over ``t`` in
#: :data:`~tests.studies._learned_rule_optimal_law.MARGIN_LEVELS` at most :data:`MAX_MARGIN`, a
#: bounded density of the blip at 0. A law that misses any of the three is a control or a
#: reporting cell.
MAX_DISAGREEMENT = 1e-3
MAX_MARGIN = 10.0

#: Each law's limit, by quadrature (:func:`tests.studies._learned_rule_optimal_law.law_limit`;
#: ``test_learned_rule_optimal_value_design`` recomputes each). ``truth`` is :math:`\Psi_{d_0}`
#: and ``bound`` the SD of :math:`D^*(d_0)`; ``treated`` is :math:`d_0`'s share; ``tie_mass``,
#: ``disagreement`` and ``margin`` are the gate's readings; ``limit_value`` is the value of the
#: law's learner's limit rule.
LIMITS: dict[str, dict[str, float]] = {
    "non_exceptional": {
        "truth": 0.577115,
        "bound": 0.664438,
        "treated": 0.549875,
        "tie_mass": 0.0,
        "disagreement": 0.0,
        "margin": 4.07,
        "limit_value": 0.577115,
    },
    "wrong_limit": {
        "truth": 0.557421,
        "bound": 0.691165,
        "treated": 0.29275,
        "tie_mass": 0.0,
        "disagreement": 0.29275,
        "margin": 1.965,
        "limit_value": 0.512179,
    },
    "partially_exceptional": {
        "truth": 0.547362,
        "bound": 0.669159,
        "treated": 0.399875,
        "tie_mass": 0.600125,
        "disagreement": 0.0,
        "margin": 2.0775,
        "limit_value": 0.547362,
    },
}


def gated(scenario: str) -> bool:
    """Whether a law's learner limit meets C3f, C7 and C8's margin reading."""
    limit = LIMITS[scenario]
    return (
        limit["tie_mass"] == 0.0
        and limit["disagreement"] <= MAX_DISAGREEMENT
        and limit["margin"] <= MAX_MARGIN
    )


#: The SD of ``sqrt(n)`` times the estimate at :math:`d_0` on ``non_exceptional`` with the known
#: ``g0``, by quadrature: RM30's bound. It sizes the noise control only, at one unit, and no
#: verdict reads an efficiency ratio.
EFFICIENCY_BOUND = LIMITS[PRIMARY_LAW]["bound"]

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The calibration label: the known-g fit of ``non_exceptional``.
CALIBRATION_LABEL = f"{PRIMARY_LAW}_known_g"
CALIBRATION_LABELS = (CALIBRATION_LABEL,)
EFFICIENCY_BOUNDS = {CALIBRATION_LABEL: EFFICIENCY_BOUND}

#: The must-fail target control (plan section 6, impl review answer 2): the known-g fit of
#: ``wrong_limit`` at :data:`CONTROL_N`, read against :math:`\Psi_{d_0}`. It passes when its
#: exact 99% coverage interval lies wholly below the calibration floor 0.92, the claim's own
#: floor, declared before the pilot. The expected coverage is about
#: ``Phi(1.96 - 3.29) = 0.09``.
TARGET_FAMILY = "optimal_target_control"
TARGET_LAW = "wrong_limit"
TARGET_LABEL = f"{TARGET_LAW}_known_g"
TARGET_CONTROL = "target_gap_control"

#: The control's n: W-f2's declared n (``tests/unit/test_learned_rule_optimal.py``), the smallest
#: multiple of 500 at which the quadrature gap 0.0452 is at least three standard errors of the
#: limit rule's curve (variance 0.4741, so 2,085 rows are needed; 2,000 gives 2.94).
CONTROL_N = 2_500

#: The estimated-g reporting cell (plan decision E2, impl review answer 1): the same law as the
#: gated cell with the declared logistic treatment learner fitted. Role ``diagnostic``, no
#: verdict; the summary publishes its bias, SE ratio and coverage of :math:`\Psi_{d_0}`.
ESTIMATED_FAMILY = "estimated_g_report"
ESTIMATED_LABEL = f"{PRIMARY_LAW}_estimated_g"
ESTIMATED = "estimated_g"

#: The partially exceptional reporting cell (impl review answer 4). Role ``diagnostic``, no
#: verdict: the contract states no claim where C7 fails, so the cell reports how the interval
#: behaves there.
EXCEPTIONAL_FAMILY = "exceptional_law_report"
EXCEPTIONAL_LAW = "partially_exceptional"
EXCEPTIONAL_LABEL = f"{EXCEPTIONAL_LAW}_known_g"
EXCEPTIONAL = "exceptional"

#: The diagnostic families, which write ``passed`` without reading a margin.
DIAGNOSTIC_FAMILIES = (ESTIMATED_FAMILY, EXCEPTIONAL_FAMILY)

PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{CALIBRATION_LABEL}__{kind}"
        for kind in (POSITIVE, "shrunken_se_control", "noise_control")
    ),
    TARGET_FAMILY: (f"{TARGET_LABEL}__{TARGET_CONTROL}",),
    ESTIMATED_FAMILY: (f"{ESTIMATED_LABEL}__{ESTIMATED}",),
    EXCEPTIONAL_FAMILY: (f"{EXCEPTIONAL_LABEL}__{EXCEPTIONAL}",),
}

#: The support row the registering change adds, declared before any run: a P12 row of the
#: fold-evaluated CV-TMLE of the value of the optimal rule with the qualifier
#: ``known_mechanism`` (impl review answer 6). Without a row declaring ``P12`` every optimal fit
#: reads experimental.
#:
#: The row is added if, and only if, these verdicts pass as declared: the primary performance
#: tests (bias leg, coverage floor, SE sanity); every ``interval_calibration`` verdict, positive
#: and both controls; and the ``optimal_target_control`` verdict.
#:
#: No row is added for estimated ``g``: its cell reports, and a row for it waits for a gated cell
#: of its own. Nothing is added for an exceptional law, where the contract claims nothing. A red
#: primary bias leg, a red positive calibration cell, or a control that does not fail is
#: diagnosed before any route is chosen. With no defect found, the cell is published red under
#: ``reporting`` with an owner row, and the row is not added.
SUPPORT_ROWS = {
    "fold_evaluated": {
        "splitting": ("fold_evaluated",),
        "families": ("P12",),
        "qualifiers": ("known_mechanism",),
    },
}

#: The truth-harness artifact: one row per primary replication.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="The value of the optimal rule",
    slug="learned-rule-optimal-value",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_optimal_value",
    document="docs/technical-reference/method-evidence/learned-rule-optimal-value.md",
    anchor="the-value-of-the-optimal-rule",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-optimal-value",
    reference=None,
    failure_bound={"cleverly-learned-rule-optimal-value": 0.01},
    modules=(
        "tests/studies/learned_rule_optimal_value.py",
        "tests/studies/learned_rule_optimal_value_properties.py",
        "tests/studies/_learned_rule_optimal_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/interventions/base.py",
        "src/cleverly/targets/builtin.py",
        "src/cleverly/learners/crossfit.py",
        "src/cleverly/learners/_fitting.py",
        "src/cleverly/_support.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_optimal_value",
    properties_module="tests.studies.learned_rule_optimal_value_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(), estimands=['ey_optimal_rule'], n_folds=10, "
        "cv_evaluation=True, targeting_scheme='pooled', g_bounds='auto', "
        "random_state=<fold seed>); every other setting the constructor's default; repeats=1"
    ),
    "known_g": (
        "fit(..., treatment_probabilities=[1 - g0(1 | W), g0(1 | W)]) with "
        "logit g0(1 | W) = 0.3 W1 - 0.2 W2, in every cell but the estimated-g one"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2); on "
        "partially_exceptional the same six products and A (W1 - 0.2)_+"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "laws": {
        "non_exceptional": (
            "RM30 law, blip 0.1 + W1; n = 2000; known g: primary and interval_calibration "
            "(SE x 0.80 and noise controls); estimated g: estimated_g_report"
        ),
        "wrong_limit": (
            "blip 3 (W1^2 - 0.5); n = 2500; known g; optimal_target_control: coverage interval "
            "upper bound < 0.92"
        ),
        "partially_exceptional": (
            "blip (W1 - 0.2)_+, zero on W1 <= 0.2; n = 2000; known g; exceptional_law_report"
        ),
    },
    "gate": (
        f"a law cell gates when its learner's limit has no tie mass (C7), differs from d0 off "
        f"the tie set on at most {MAX_DISAGREEMENT} (C3f), and max_t P(0 < |b0| <= t) / t <= "
        f"{MAX_MARGIN} (C8, exponent 1); only non_exceptional gates"
    ),
    "variance": (
        "V^-2 sum_v n_v^-2 sum_{i in v} D_i^2 of the raw fold curves, uncentred within each "
        "fold (JCI App B)"
    ),
    "truth": (
        "Psi_d0 = E Qbar0(d0(W), W), d0 = 1{Qbar0(1, W) > Qbar0(0, W)}, by the trapezoid rule on "
        "4001 points of W1 for each W2; one number per law"
    ),
    "regret": (
        "per replication: refit each fold's outcome learner on the fit's training rows with "
        "the package's fit_on_rows, require the fit's rule on every validation row (rule L4), "
        "integrate each fold rule's value by the same rule, and publish rule_value = (1/V) sum_v "
        "Psi_{d_v} and regret = Psi_d0 - rule_value; descriptive"
    ),
    "rule_disagreement": "the record's share of rows on which the ten fold rules disagree",
    "oracle_se": "sqrt(V^-2 sum_v Var D*(d_v) / n_v) at the fold rules; descriptive",
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed, with the fixed truth :math:`\\Psi_{d_0}`."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(scenario, n, seed), {law.ESTIMAND: law.law_limit(scenario)["truth"]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, scenario, law.LEARNERS[scenario](), fold_seed, known_g=True)
    estimate = fitted.result.estimates[law.ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    truth = law.law_limit(scenario)["truth"]
    return [
        {
            "implementation": record.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": law.ESTIMAND,
            "truth": truth,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= truth <= high),
            "initial_estimate": base.untargeted_estimate(fitted.result),
            **law.harness_columns(fitted, scenario),
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.draw(scenario, size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``, with its harness columns.

    ``record`` is the declared one, or the throwaway record of a smoke run (rule L9).
    """
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, n)
            for scenario in record.scenarios
            for replicate in range(replicates)
        ],
        partial(_primary, record),
        draw=partial(_sample, record),
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *law.HARNESS_COLUMNS]])


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication."""
    columns = ["scenario", "replicate", "truth", *law.HARNESS_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "replicate"], ignore_index=True
        )
    }
