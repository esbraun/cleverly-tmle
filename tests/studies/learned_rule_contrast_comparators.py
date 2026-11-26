r"""Declared evidence for the learned rule against stochastic regimes and the natural course (T58).

The subject is ``LearnedRuleContrast`` with a known ``Stochastic`` comparator or the natural
course, through its engine form ``TMLE(learned_rule=LearnedRule(), interventions=[...],
n_folds=10, targeting_scheme="pooled", g_bounds="auto")`` with RM30's interaction logistic
outcome learner and logistic treatment learner. A static comparator and a known ``Rule`` are
T44's (the typed route is the engine fit bit for bit, and a known rule reduces bit for bit to a
one-hot column). A non-one-hot ``g*/g`` changes the comparator's clever covariate, its variance
and its covariance with the learned column, and the natural course is a column outside the
regime set, so neither is covered by a bit-for-bit reduction.

Each draw is fitted twice on one fold seed, under both reports
(:mod:`tests.studies._learned_rule_contrast_comparators_law`):

- **fit C** beside ``[Static(0), coin, offer, NaturalCourse()]``. ``coin`` treats with
  probability one half; ``offer`` treats with probability one half where ``W1 > 0`` and never
  elsewhere. The natural course never enters the joint fluctuation, so fit C's learned and
  regime columns are those of ``[Static(0), coin, offer]`` bit for bit, which lets fit C's
  stochastic cells stand for that configuration;
- **fit N** beside ``[NaturalCourse()]`` alone, the natural-course-only call.

Two laws:

``non_exceptional``
    RM30's correctly specified law at n = 2,000: the calibration bands.
``misspecified_limit``
    RM30's law whose blip the outcome model cannot represent, with the correct logistic
    treatment model fitted by maximum likelihood. The contrasts against ``Static(0)``,
    ``coin`` and ``offer`` share the fitted ``g`` with the learned column, so the package's
    projection step makes their reported variance conservative. The natural-course contrast does
    not: the natural course reads no ``g``, so the reported variance misses
    ``Var(P) - 2 Cov(D_N, P)`` with ``P`` the projection of ``D_L`` on the treatment model's
    scores, a term of either sign. The design test predicts it as a reported over true SE of
    about 0.998 on this law, which the coverage cells cannot tell from 1. These cells measure
    coverage under a misspecified outcome model, not conservativeness.

Every quantity here was declared before any run, and no declared run has been made.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_contrast_comparators_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

base = law.base

#: The primary replications: the fold-evaluated fits of ``non_exceptional``, all eleven labelled
#: parameters (fit C's nine and fit N's learned value and contrast), each held to the bias leg,
#: the coverage floor and the SE sanity band. 1,600 puts each primary coverage cell's chance of
#: a false red at or below 0.002 at a realized coverage of 0.94
#: (``test_learned_rule_contrast_comparators_design``).
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20265809
RESAMPLING_SEED = 20265810

PRIMARY_LAW = "non_exceptional"
SCENARIOS = {PRIMARY_LAW: tuple(law.LABELS.values())}

#: The two reports, in the order every cell name reads them.
REPORTS = tuple(law.REPORTS)

#: The positives of the calibration family: the comparators and contrasts T44 does not measure.
#: The learned value and ``Static(0)`` are calibrated in T44 and are not re-gated. The ``natural``
#: cell checks the variance arithmetic of a mean and is not contrast evidence.
CALIBRATION_TAGS = (
    "coin",
    "offer",
    "natural",
    "contrast_0",
    "contrast_coin",
    "contrast_offer",
    "contrast_natural",
    "alone_contrast_natural",
)

#: The SD of ``sqrt(n)`` times each calibration tag's estimate at the limit rule of the
#: declared learner's population projection on ``non_exceptional``, by quadrature
#: (:func:`~tests.studies._learned_rule_contrast_comparators_law.efficiency_bounds`). They size
#: the noise controls only, at one unit each. ``test_learned_rule_contrast_comparators_design``
#: recomputes them, and checks the ``natural`` bound against ``sqrt(mu0 (1 - mu0))``.
TAG_BOUNDS = {
    "coin": 0.497895,
    "offer": 0.585860,
    "natural": 0.499294,
    "contrast_0": 0.708845,
    "contrast_coin": 0.479627,
    "contrast_offer": 0.403690,
    "contrast_natural": 0.454565,
    "alone_contrast_natural": 0.454565,
}

#: The positive arm's cell suffix.
POSITIVE = "declared"

CALIBRATION_LABELS = tuple(
    f"{PRIMARY_LAW}_{report}_{tag}" for report in REPORTS for tag in CALIBRATION_TAGS
)
EFFICIENCY_BOUNDS = {
    f"{PRIMARY_LAW}_{report}_{tag}": TAG_BOUNDS[tag]
    for report in REPORTS
    for tag in CALIBRATION_TAGS
}

#: The ``misspecified_limit`` family: each report's contrasts, held to the bias leg and to a
#: coverage interval whose lower bound is at or above the calibration floor 0.92, with no
#: SE-ratio gate. Each positive's control is the estimate with its SE times
#: :data:`COVERAGE_SHRINK`, whose coverage interval must fall wholly below 0.92.
COVERAGE_FAMILY = "conservative_coverage"
COVERAGE_LAW = "misspecified_limit"
COVERAGE_TAGS = law.CONTRAST_TAGS
COVERAGE_LABELS = tuple(
    f"{COVERAGE_LAW}_{report}_{tag}" for report in REPORTS for tag in COVERAGE_TAGS
)
COVERAGE_CONTROL = "shrunken_coverage_control"
COVERAGE_SHRINK = 0.60

#: The correlation-one control of each contrast, both laws and reports: the SE
#: ``|se_L - se_K|``, whose coverage interval must lie wholly below
#: :data:`CORRELATION_ONE_CEILING` (T44's must-fail convention).
COVARIANCE_FAMILY = "contrast_covariance"
CORRELATION_ONE = "correlation_one_control"
CORRELATION_ONE_CEILING = 0.80

#: The independent-SE diagnostic of each contrast: role ``diagnostic``, no verdict.
INDEPENDENT_FAMILY = "independent_se_diagnostic"
INDEPENDENT = "independent_se"

#: The bias control: fit C with RM30's ``drop_w1_learner``, fold-evaluated, read on the
#: contrast against ``offer``. The learner's limit rule treats every row, and the offer
#: contrast's limit plug-in bias is about three SEs (the design test).
NECESSITY_FAMILY = "targeting_necessity"
NECESSITY_TAG = "contrast_offer"
NECESSITY_LABEL = f"drop_w1_{NECESSITY_TAG}"
NECESSITY_CELLS = (f"{NECESSITY_LABEL}__targeted", f"{NECESSITY_LABEL}__untargeted")


def covariance_cells(scenario: str) -> tuple[str, ...]:
    """The correlation-one control cells of one law."""
    return tuple(
        f"{scenario}_{report}_{tag}__{CORRELATION_ONE}"
        for report in REPORTS
        for tag in law.CONTRAST_TAGS
    )


def independent_cells(scenario: str) -> tuple[str, ...]:
    """The independent-SE diagnostic cells of one law."""
    return tuple(
        f"{scenario}_{report}_{tag}__{INDEPENDENT}"
        for report in REPORTS
        for tag in law.CONTRAST_TAGS
    )


PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in (POSITIVE, "shrunken_se_control", "noise_control")
    ),
    COVERAGE_FAMILY: tuple(
        f"{label}__{kind}" for label in COVERAGE_LABELS for kind in (POSITIVE, COVERAGE_CONTROL)
    ),
    COVARIANCE_FAMILY: tuple(cell for s in law.SCENARIOS for cell in covariance_cells(s)),
    INDEPENDENT_FAMILY: tuple(cell for s in law.SCENARIOS for cell in independent_cells(s)),
    NECESSITY_FAMILY: NECESSITY_CELLS,
}

#: The tags each row's verdicts read. ``combined`` is fit C's key and reads all of fit C;
#: ``stochastic`` is the key of the learned rule beside known regimes with at least one
#: stochastic one, which fit C's regime columns measure literally; ``natural_course`` is fit N's
#: key, whose ``natural`` cell is fit C's number.
ROW_TAGS = {
    "combined": tuple(tag for tag, (fit, _) in law.TAGS.items() if fit == "combined"),
    "stochastic": (
        "value",
        "known_0",
        "coin",
        "offer",
        "contrast_0",
        "contrast_coin",
        "contrast_offer",
    ),
    "natural_course": ("alone_value", "natural", "alone_contrast_natural"),
}

#: The support rows the registering change adds, declared before any run: three rows per
#: report, each with its exact qualifier set. A row covers every parameter a fit with its key
#: reports, because a row labels every estimate.
#:
#: A row of a report is added if, and only if, these verdicts pass as declared: the primary
#: performance tests of its tags; every ``interval_calibration`` verdict of its tags under that
#: report, positives and controls; every ``conservative_coverage`` and ``contrast_covariance``
#: verdict of its contrast tags under that report, both laws; and, for ``combined`` and
#: ``stochastic``, the ``targeting_necessity`` verdicts. The primary and necessity cells are
#: fold-evaluated; at ten equal folds without weights the stacked point is the fold-evaluated
#: point, and only the variance rules differ, as in T44.
#:
#: Reading a red cell, declared before the run. (i) When diagnosis finds a defect, the row it
#: gates is not added and the labels it covers stay experimental. (ii) When the 99% interval of
#: a red calibration cell's SE ratio contains 1, its coverage is consistent with 0.95 and
#: diagnosis finds no defect, the red is a finite-sample one: the row goes to ``reporting`` with
#: an owner in the roadmap's red-cell owners table. A red cell is not re-run on new seeds. A
#: control that does not fail is a design defect.
SUPPORT_ROWS = {
    f"{row}.{report}": {"splitting": (report,), "qualifiers": qualifiers}
    for row, qualifiers in (
        ("combined", ("known_regimes", "known_stochastic_regimes", "natural_course")),
        ("stochastic", ("known_regimes", "known_stochastic_regimes")),
        ("natural_course", ("known_regimes", "natural_course")),
    )
    for report in REPORTS
}

#: What each row's limits say beyond T44's, declared with the rows.
ROW_LIMITS = {
    "stochastic": (
        "the misspecified-law cells measure coverage only; the shared-g projection term is "
        "predicted negligible on that law (test_learned_rule_contrast_comparators_design)"
    ),
    "natural_course": (
        "with a misspecified outcome model and a maximum-likelihood g there is no "
        "conservativeness result: the reported variance misses Var(P) - 2 Cov(D_N, P), of either "
        "sign, predicted about 0.998 as an SE ratio on misspecified_limit, which the coverage "
        "cells cannot tell from 1. A natural course beside a static or rule regime shares this "
        "key without being fitted; its learned column moves through the joint fluctuation by "
        "about 0.05 SE, and the row rests that composition on the stacking argument"
    ),
}

#: The truth-harness artifact: one row per primary replication and parameter.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned-rule contrasts with stochastic regimes and the natural course",
    slug="learned-rule-contrast-comparators",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_contrast_comparators",
    document="docs/technical-reference/method-evidence/learned-rule-contrast-comparators.md",
    anchor="learned-rule-contrasts-with-stochastic-regimes-and-the-natural-course",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-contrast-comparators",
    reference=None,
    failure_bound={"cleverly-learned-rule-contrast-comparators": 0.01},
    modules=(
        "tests/studies/learned_rule_contrast_comparators.py",
        "tests/studies/learned_rule_contrast_comparators_properties.py",
        "tests/studies/_learned_rule_contrast_comparators_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/interventions/base.py",
        "src/cleverly/targets/base.py",
        "src/cleverly/targets/builtin.py",
        "src/cleverly/learners/crossfit.py",
        "src/cleverly/_support.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_contrast_comparators",
    properties_module="tests.studies.learned_rule_contrast_comparators_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit_combined": (
        "TMLE(learned_rule=LearnedRule(), interventions=[Static(0), Stochastic(coin), "
        "Stochastic(offer), NaturalCourse()], n_folds=10, cv_evaluation=True, "
        "targeting_scheme='pooled', g_bounds='auto', random_state=<fold seed>) at n = 2000; "
        "coin treats with probability 0.5, offer with probability 0.5 where W1 > 0; every other "
        "setting the constructor's default"
    ),
    "fit_natural_course": (
        "the same with interventions=[NaturalCourse()] alone, on the same draw and fold seed"
    ),
    "stacked_fits": "each fit again with cv_evaluation=False, on the same draw and fold seed",
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "parameters": dict(law.LABELS),
    "laws": {
        "non_exceptional": "RM30 law with blip 0.1 + W1; calibration bands",
        "misspecified_limit": (
            "RM30 law with blip 0.8 W1 + W1^2 - 0.3; bias leg and coverage floor 0.92 only; "
            "the cells measure coverage, not conservativeness"
        ),
    },
    "truth": (
        "per replication: refit each fold's rule (rule L4) and integrate its value on 4001 "
        "points of W1 for each W2, weighted by the report's fold weights; a known regime's "
        "truth is E sum_a g*(a | W) Qbar0(a, W) and the natural course's E Y, by the same rule; "
        "a contrast's is the difference"
    ),
    "oracle_se": "sqrt(sum_v omega_v^2 Var D_v / n_v) of each curve at the fold rules; descriptive",
    "conservative_coverage": (
        "misspecified_limit: bias leg and coverage interval lower bound >= 0.92; control: SE x "
        "0.60, coverage interval upper bound < 0.92"
    ),
    "contrast_covariance": (
        "each contrast with se = |se_L - se_K| (correlation one); coverage interval upper "
        "bound < 0.80"
    ),
    "independent_se_diagnostic": (
        "each contrast with se = sqrt(se_L^2 + se_K^2) (independence); diagnostic, no verdict"
    ),
    "targeting_necessity": (
        "RM30's drop_w1_learner, fit C fold-evaluated, contrast against offer: targeted against "
        "the untargeted difference of the initial plug-ins"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return base.draw(scenario, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def harness_columns(fitted: law.Measured, oracle: float) -> dict[str, Any]:
    """The harness columns of one row of a draw's fits."""
    return {
        "oracle_se": oracle,
        "rule_rows_checked": fitted.refit.rows_checked,
        "solver_warnings": fitted.solver_warnings,
    }


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, scenario, base.outcome_learner(), fold_seed, ("fold_evaluated",))
    truths = law.truths(fitted.result("value", "fold_evaluated"), fitted.refit, scenario)
    rows = []
    for tag, (_, name) in law.TAGS.items():
        result = fitted.result(tag, "fold_evaluated")
        estimate = result.estimates[name]
        std_error, low, high = reported_inference(estimate)
        truth, oracle = truths[tag]
        rows.append(
            {
                "implementation": record.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": law.LABELS[tag],
                "truth": truth,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": law.untargeted(result, tag),
                **harness_columns(fitted, oracle),
            }
        )
    return rows


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return base.draw(scenario, size, replicate_seed(record, scenario, replicate))


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
    """The harness columns of every primary replication and parameter."""
    columns = ["scenario", "replicate", "estimand", "truth", *law.HARNESS_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "replicate", "estimand"], ignore_index=True
        )
    }
