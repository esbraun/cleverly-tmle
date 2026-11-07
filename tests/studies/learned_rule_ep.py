"""Declared evidence for the learned-rule value with the EP rule learner (P1 part (c)).

The subject is ``TMLE(learned_rule=LearnedRule(loss="ep", modifiers=Modifiers(["W1", "W2"],
domains={"W1": (-1, 1)}, levels={"W2": (0, 1)})), n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto")``.  Each fold's rule is the sign of the EP curve of
the effect given ``(W1, W2)`` (van der Laan, Carone and Luedtke, arXiv:2402.01972v2), fitted on
the fold's training rows from the shipped 5-fold inner cross-fit, with the default sieve's size
chosen by cross-validation over the inner folds and the nuisances refitted inside each.  The
outer fluctuation, estimate and variance are the plug-in rule's.  The scenarios are the RM30
laws ``non_exceptional`` and ``misspecified_limit`` with the main-terms outcome learner and the
logistic treatment learner, rows only: no weights, no clusters, pooled targeting, one split.

The primary phase fits n = 2,000, 2,000 replications per law, read by the framework defaults
(the coverage floor 0.90 and the SE sanity screen).  The gated calibration cells fit n = 16,000
(:mod:`tests.studies.learned_rule_ep_properties`, plan rev 2.6), the size at which the size
pilot finds the error variance of the learned-value fits within about 1.5% (one SE) of that of
a fit whose rule is fixed.

The target of each replication is data-adaptive, so each row carries its own truth: the
fold-average value of the rules the fit learned (rule L4).  :mod:`tests.studies._learned_rule_ep_law`
holds the fits, the limits and the controls.  The P1 plan (``P1-plan.md`` rev 2.7 in the plans
repository, section 5.5 and its amendments) declares every quantity here before any run, and
the sizing pilots (:mod:`tests.studies.learned_rule_ep_pilot` and
:mod:`tests.studies.learned_rule_ep_size_pilot`) ran on streams no declared draw reads.

The study is registered in :func:`tests.studies.evidence.registry.registered` with the artifacts
of its declared run, which the fast tier reads.  Its support rows are ``T51`` and
``T51.optimal``: the plan's ``T49`` rows, renumbered because the ``conditional-projection``
rows took ``T49`` and ``T50`` first.  Every name ``T49`` in this study's modules and in the
declared run's README means those rows.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_ep_law as law
from tests.studies import _learned_rule_law as base
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 2_000
PRIMARY_N = 2_000
#: The record seeds: the first candidate from 20261141 in steps of 10, skipping the seeds another
#: study declares, whose sample seeds meet no other learned-rule study's and no pilot seed, whose
#: fold seeds meet no pilot seed, and whose property labels need no rule-L3 retry; checked on
#: the seeds alone (``tests/unit/test_learned_rule_ep_design.py``).
SEED = 20261481
RESAMPLING_SEED = 20261482

SCENARIOS = dict.fromkeys(law.LAWS, (law.ESTIMAND,))

#: The asymptotic SD of ``sqrt(n)`` times the estimate at each calibration label's limit rule,
#: by the RM30 grid quadrature: the EP limit rule of each law
#: (:func:`~tests.studies._learned_rule_ep_law.limit_value`), and ``d0`` for the optimal value.
#: They size the noise controls only, at one unit each; no verdict reads an efficiency ratio.
EFFICIENCY_BOUNDS = {
    "non_exceptional": 0.664747,
    "misspecified_limit": 0.663107,
    "optimal_known_g": 0.664438,
}

#: The calibration kinds, in the order the rows are written.
CALIBRATION_KINDS = ("declared", "shrunken_se_control", "noise_control")

#: The fold-locality control, a cell of its own at the primary size (plan 5.5).
CONTROL_LABEL = "non_exceptional"
CONTROL_KIND = "validation_rule_control"

#: The reporting families: n = 1,000 per gated law, and the exceptional law at n = 2,000.
DIAGNOSTIC_FAMILY = "small_sample_diagnostic"
DIAGNOSTIC_KIND = "n1000"
EXCEPTIONAL_FAMILY = "exceptional_law_report"
EXCEPTIONAL_CELL = f"{law.EXCEPTIONAL_LAW}__n2000"

CALIBRATION_LABELS = (*law.LAWS, "optimal_known_g")

PROPERTY_CELLS = {
    "interval_calibration": (
        *(f"{label}__{kind}" for label in CALIBRATION_LABELS for kind in CALIBRATION_KINDS),
        f"{CONTROL_LABEL}__{CONTROL_KIND}",
    ),
    DIAGNOSTIC_FAMILY: tuple(f"{label}__{DIAGNOSTIC_KIND}" for label in law.LAWS),
    EXCEPTIONAL_FAMILY: (EXCEPTIONAL_CELL,),
}

HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned-rule value, EP rule learner",
    slug="learned-rule-ep",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_ep",
    document="docs/technical-reference/method-evidence/learned-rule-ep.md",
    anchor="learned-rule-value-ep-rule-learner",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-ep",
    reference=None,
    failure_bound={"cleverly-learned-rule-ep": 0.01},
    modules=(
        "tests/studies/learned_rule_ep.py",
        "tests/studies/learned_rule_ep_properties.py",
        "tests/studies/_learned_rule_ep_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/_learned_rule_optimal_law.py",
        "src/cleverly/interventions/_rule_learners.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/estimators/ep.py",
        "src/cleverly/effect_modification.py",
        "src/cleverly/bases.py",
        "src/cleverly/estimators/tmle.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_ep",
    properties_module="tests.studies.learned_rule_ep_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(loss='ep', modifiers=Modifiers(['W1', 'W2'], "
        "domains={'W1': (-1, 1)}, levels={'W2': (0, 1)})), n_folds=10, cv_evaluation=True, "
        "targeting_scheme='pooled', g_bounds='auto', random_state=<fold seed>); primary "
        "n = 2000, interval_calibration n = 16000 (the fold-locality control n = 2000), "
        "small_sample_diagnostic n = 1000, exceptional_law_report n = 2000; rows only, repeats=1"
    ),
    "rule_learner": (
        "the EP curve on the default sieve Additive(Cosine(['W1'], 6), Indicators('W2')), its "
        "size 1 to 6 chosen by the one-step risk over the 5 inner folds with the nuisances "
        "refitted in 5 folds of each inner training fold; least squares final stage"
    ),
    "optimal_value": (
        "the same fit with estimands=['ey_optimal_rule'] and treatment_probabilities=g0 at "
        "non_exceptional; truth Psi_{d0}"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "laws": "RM30 non_exceptional and misspecified_limit; exceptional (reporting)",
    "truth": (
        "per replication: refit each fold's rule through fit_fold_rule with the recorded inner "
        "seed, the fit's g_bounds and alpha, require the fit's rule on every validation row, "
        "and average the fold rules' values E Qbar0(d, W) by the trapezoid rule on 4001 points "
        "of W1 for each W2"
    ),
    "value_gain": (
        "plug_in_estimate, plug_in_std_error and plug_in_truth: the plug-in rule's fit of the "
        "same sample with the same fold seed; descriptive"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate); "
    "the inner seeds derive from it by SeedSequence([split_seed, v])",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return base.draw(scenario, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, scenario, fold_seed)
    estimate = fitted.result.estimates[law.ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    return [
        {
            "implementation": record.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": law.ESTIMAND,
            "truth": fitted.truth.truth,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= fitted.truth.truth <= high),
            "initial_estimate": base.untargeted_estimate(fitted.result),
            "oracle_se": fitted.truth.oracle_se,
            "rule_rows_checked": fitted.truth.rows_checked,
            "solver_warnings": fitted.warnings,
            **law.value_gain_columns(fitted),
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return base.draw(scenario, size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``, with its harness columns."""
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
    return replication.result(
        outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *base.HARNESS_COLUMNS, *law.VALUE_GAIN_COLUMNS]]
    )


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness and value-gain columns of every primary replication."""
    columns = ["scenario", "replicate", "truth", *base.HARNESS_COLUMNS, *law.VALUE_GAIN_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "replicate"], ignore_index=True
        )
    }
