r"""Registered evidence for the in-sample learned-rule value (X11 PR F2, part (a)).

The subject is ``TMLE(learned_rule=LearnedRule(), cross_fit=False, g_bounds="auto")`` with the
declared outcome learner :class:`~tests.studies._learned_rule_in_sample_law.InteractionLogistic`
(RM30's interaction logistic regression, marked parametric) and RM30's logistic treatment
learner. It learns one plug-in rule :math:`d_n` from the outcome regression fitted on every row,
and runs the shipped in-sample regime TMLE at :math:`d = d_n`: one fluctuation solving
:math:`P_n D^*(d_n, \bar Q^*_n, g_n) = 0`, the plug-in estimate, the curve
:math:`\phi_i = Z_i - \hat\psi` with :math:`Z_i = H_i(Y_i - \bar Q^*_i) + \bar Q^*_i(d_n)`, and
the stored variance the centred rule :math:`\{n(n-1)\}^{-1}\sum_i(\phi_i - \bar\phi)^2`, with
:math:`\bar\phi = 0` to solver tolerance (a factor :math:`n/(n-1) = 1.0005` over
:math:`n^{-2}\sum_i\phi_i^2` at n = 2,000).

The target of each replication is data-adaptive, :math:`\Psi_{d_n}(P_0)`, the value of the one
rule learned on every row (van der Laan and Luedtke 2015, Section 6, Theorem 5, conditions
C3a, C4a and C5a of the learned-rule contract), so each row carries its own truth. Two laws:

``non_exceptional``
    RM30's correctly specified law at n = 2,000.
``misspecified_limit``
    RM30's law whose blip the declared outcome model cannot represent, at n = 2,000. The rule
    converges to the rule of the model's projection, and the correct treatment model carries
    the remainder.

The X11 wave-2 plan (``X11-wave2-plan.md`` rev 2.1 in the plans repository, section 3), its
review (finding 4, Q4, R1, R2), the implementation review ``reviews/x11/X11-F2-impl-review.md``
and the study design record ``reviews/x11/X11-F2-study-design.md`` declare every quantity here
before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_in_sample_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The primary replications of each law. The primary gate reads the bias leg (the 99% interval
#: of the mean error inside 0.25 error SDs), and the in-sample estimator carries an optimism bias
#: of about +0.12 (``non_exceptional``, 4,000 pilot draws) and +0.21 (``misspecified_limit``,
#: 8,000 pilot and review-probe draws pooled) error SDs at n = 2,000. 14,400 is the smallest
#: multiple of 400 whose predictive pass probability of both bias legs is at least 0.95 (0.9529;
#: 14,000 gives 0.9498), by the rule of ``test_learned_rule_in_sample_design``. The design
#: record states why (D3) and why the count moved from 8,400 before the run (D6).
PRIMARY_REPLICATES = 14_400
PRIMARY_N = 2_000
SEED = 20263081
RESAMPLING_SEED = 20263082

SCENARIOS = dict.fromkeys(law.SCENARIOS, (law.ESTIMAND,))

#: The SD of the row curve ``D*(d_1, Qbar0, g0)`` at each law's limit rule, the rule of the
#: declared learner's population projection (:func:`~tests.studies._learned_rule_in_sample_law.
#: efficiency_bound`). They equal the ``learned-rule-compositions`` ``fold_targeted`` bound and
#: the ``learned-rule-split-mean`` ``misspecified_limit`` bound of the same laws and learner.
#: They size the noise controls only, at one unit each, and no verdict reads an efficiency ratio.
#: ``test_learned_rule_in_sample_design`` recomputes them.
EFFICIENCY_BOUNDS = {
    "non_exceptional": 0.664438,
    "misspecified_limit": 0.666347,
}

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The ``non_exceptional`` cell's own control: the interpolating tree, clipped to [0.01, 0.99],
#: fitted in sample on the same draw (plan section 3, "Controls that must fail"). It is
#: adaptive, so condition C5a fails, and its curve understates the spread of its error: its
#: SE-ratio interval must fall below the band. Its truth is the value of its own rule.
INTERPOLATING = "interpolating_control"

#: The reporting family: the declared estimate and interval of each law read against the
#: fold-average target of the same draw's 10-fold rules, with the role ``diagnostic``. No margin
#: is read and no verdict forms. It measures how far the in-sample target sits from the
#: cross-fitted one at this size, which no calibration verdict can see. Both laws carry a cell:
#: on ``misspecified_limit`` the limit rule is not the optimal one, so the value has a
#: first-order sensitivity to the rule's boundary (the design record, review fix 7).
REPORTING_FAMILY = "target_distinction"
REPORTING_CELLS = tuple(f"{label}__fold_average" for label in law.SCENARIOS)

#: The weak-margin diagnostic of the bias control (the design record, D2): the plan's
#: ``drop_w1_learner`` (marked parametric), whose limit blip is 0.037 and 0.049 on the
#: probability scale, too small at n = 2,000 for its one rule to settle (condition C3a fails at
#: this n). On the exact law its rule differs from its limit rule in about 17.5% of samples.
#: Its targeted and untargeted arms are published with the role ``diagnostic``, no margin and
#: no verdict, beside the declared ``targeting_necessity`` family, whose learner has a strong
#: limit blip. The summary publishes the share of draws whose rule flipped and the targeted
#: arm's bias on settled and on flipped draws.
WEAK_LIMIT_FAMILY = "weak_limit_necessity"
WEAK_LIMIT_LABEL = "non_exceptional_drop_w1"
WEAK_LIMIT_CELLS = (f"{WEAK_LIMIT_LABEL}__targeted", f"{WEAK_LIMIT_LABEL}__untargeted")

#: The value on ``non_exceptional`` of the ``drop_w1`` limit rule, which treats every row (its
#: limit blip is positive at both values of ``W2``). A ``drop_w1`` rule is constant in ``W1``,
#: so it takes one of four values, and a draw's truth differs from this one exactly when its
#: rule flipped at some level of ``W2`` (``test_learned_rule_in_sample_design``).
WEAK_LIMIT_RULE_VALUE = 0.5316158567592173


PROPERTY_CELLS = {
    "interval_calibration": (
        *tuple(
            f"{label}__{kind}"
            for label in law.SCENARIOS
            for kind in (POSITIVE, "shrunken_se_control", "noise_control")
        ),
        f"non_exceptional__{INTERPOLATING}",
    ),
    "targeting_necessity": ("non_exceptional__targeted", "non_exceptional__untargeted"),
    REPORTING_FAMILY: REPORTING_CELLS,
    WEAK_LIMIT_FAMILY: WEAK_LIMIT_CELLS,
}

#: The support row the registering change adds if, and only if, these verdicts pass as declared:
#: the primary performance tests of both laws (bias leg, coverage floor, SE sanity), every
#: ``interval_calibration`` verdict, and the ``targeting_necessity`` verdicts. The row is family
#: P11, ``splitting`` ``in_sample``, with the key options below, exactly. A red primary bias leg
#: or a red positive calibration cell is diagnosed as finite-sample in-sample optimism, not a
#: defect, unless the investigation finds one; the cell is published red under ``reporting``
#: with an owner row, and no P11 row is added (the design record, "Support row"). It covers the
#: plug-in rule with a parametric outcome and a parametric treatment learner, where C5a holds.
#: No cell fits a rule learner or supplies ``treatment_probabilities=``, so the row declares no
#: ``rule_learner_family`` and no ``known`` treatment family, and those fits stay experimental
#: (impl review open question 2).
SUPPORT_OPTIONS = {
    "outcome_learner_family": "parametric",
    "treatment_learner_family": "parametric",
}

#: The truth-harness artifact: one row per primary replication.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="In-sample learned-rule value",
    slug="learned-rule-in-sample",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_in_sample",
    document="docs/technical-reference/method-evidence/learned-rule-in-sample.md",
    anchor="in-sample-learned-rule-value",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-in-sample",
    reference=None,
    failure_bound={"cleverly-learned-rule-in-sample": 0.01},
    modules=(
        "tests/studies/learned_rule_in_sample.py",
        "tests/studies/learned_rule_in_sample_properties.py",
        "tests/studies/_learned_rule_in_sample_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/learners/crossfit.py",
        "src/cleverly/_support.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_in_sample",
    properties_module="tests.studies.learned_rule_in_sample_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(), cross_fit=False, cv_evaluation=False, "
        "g_bounds='auto', random_state=<fit seed>) at n = 2000; repeats=1"
    ),
    "outcome_learner": (
        "InteractionLogistic: PolynomialFeatures(2, interaction_only=True, include_bias=False) "
        "then LogisticRegression(C=1e6, max_iter=5000, tol=1e-10) on (A, W1, W2), in a class "
        "marked cleverly_learner_family = 'parametric'"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "support_key": (
        "splitting ('in_sample',), outcome_learner_family 'parametric', "
        "treatment_learner_family 'parametric', checked on every fit"
    ),
    "laws": {
        "non_exceptional": "RM30 law with blip 0.1 + W1",
        "misspecified_limit": "RM30 law with blip 0.8 W1 + W1^2 - 0.3",
    },
    "variance": (
        "{n (n - 1)}^-1 sum_i (phi_i - phibar)^2, phibar = 0 to solver tolerance; "
        "the factor n / (n - 1) = 1.0005 over n^-2 sum_i phi_i^2"
    ),
    "truth": (
        "per replication: refit the outcome learner on every row with the package's "
        "fit_on_rows, require the fit's rule on every row (rule L4), and integrate the rule's "
        "value by the trapezoid rule on 4001 points of W1 for each W2, with the known Qbar0"
    ),
    "oracle_se": "sqrt(Var_P0 D*(d_n, Qbar0, g0) / n) at the fit's rule; descriptive",
    "interpolating_control": (
        "DecisionTreeClassifier(min_samples_leaf=1, random_state=0) with predict_proba "
        "clipped to [0.01, 0.99], no family marker, fitted in sample on the same draw"
    ),
    "targeting_necessity": (
        "WithoutW1MainLogistic: logistic on (A, W2, A W1, A W2), marked parametric; the "
        "control is the same fit's initial plug-in mean of Qbar(d_n(W_i), W_i)"
    ),
    "weak_limit_necessity": (
        "the same pair with DropW1Logistic (RM30's drop_w1_learner, marked parametric), "
        "whose limit blip is weak at n = 2000; diagnostic, no verdict; the summary publishes "
        "the share of draws whose rule differs from the limit rule (truth != "
        "WEAK_LIMIT_RULE_VALUE) and the targeted arm's mean error on settled and on flipped "
        "draws, in the targeted arm's error SD"
    ),
    "target_distinction": (
        "the declared estimate and interval of each law read against (1/10) sum_v "
        "Psi_{d_v}(P0), d_v the declared learner's plug-in rule on the training rows of "
        "make_folds(n, 10, random_state=<fold seed>); descriptive"
    ),
    "fit_seed": "stream_seed(record, 'fit_seed', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.base.draw(scenario, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fit_seed = stream_seed(record, "fit_seed", "primary", scenario, replicate)
    result, truth, counts = law.measured(frame, scenario, "declared", fit_seed)
    estimate = result.estimates[law.ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    return [
        {
            "implementation": record.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": law.ESTIMAND,
            "truth": truth.truth,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= truth.truth <= high),
            "initial_estimate": law.untargeted_estimate(result),
            **law.harness_columns(truth, counts),
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.base.draw(scenario, size, replicate_seed(record, scenario, replicate))


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
