"""Registered evidence for the learned-rule value inside baseline strata (X11 PR E2, part (m)).

The subject is ``TMLE(learned_rule=LearnedRule(), cross_fit=True, n_folds=5,
cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto", simultaneous=False)`` fitted
with ``strata=["S"]`` on complete data with rows as the unit and ``repeats=1``: with
``q_bounds=(0, 1)`` on a proportion outcome and without ``q_bounds`` on a binary one. Fold
``v``'s plug-in rule ``d_v`` comes from its training rows. Each stratum estimate is the ``1/V``
average of the fold plug-ins at ``d_v`` over the rows of each validation fold in the stratum,
each fold's stratum curve divides by the fold's own stratum share, and the marginal estimate is
the share mixture of the stratum estimates. Each fit reports four parameters: the learned-rule
value ``ey_learned_rule[learned rule]`` in each of the three strata and marginally.

``unweighted``, ``weighted``
    The bounded (beta) outcome with the blip ``0.1 + 1.5 W1``, whose rule boundary lies in
    stratum 1; rows only, and with the known weight ``w = 1 + 1.5 W2``.
``binary_steep``, ``binary_steep_weighted``
    The binary outcome with the blip ``-1.2 + 3.0 W1``, whose rule boundary lies in stratum 2.

Every setting draws the law of :mod:`tests.studies._fold_evaluated_strata_law` (X11 PR E1),
read and never edited, with a steeper blip than E1's in the binary settings
(:func:`tests.studies._learned_rule_strata_law.draw`). The target of each replication is data-adaptive, so each row carries
its own truth: the fold average of the stratum values of the rules the fit learned, refit and
checked by :func:`tests.studies._learned_rule_strata_law.harness`. The X11 plan
(``X11-plan.md`` rev 2.1 in the plans repository, sections 2.6 and 3), the implementation notes
``reviews/x11/X11-E2-impl-notes.md`` and the study design record
``reviews/x11/X11-E2-study-design.md`` declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_strata_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 2_000
PRIMARY_N = 1_000
SEED = 20263061
RESAMPLING_SEED = 20263062

SCENARIOS = dict.fromkeys(law.SETTINGS, law.PARAMETERS)

#: The calibration labels of each setting: one per parameter, ``<setting>_<stratum>``.
SETTING_LABELS = {
    setting: tuple(f"{setting}_{label}" for label in law.PARAMETER_LABELS)
    for setting in law.SETTINGS
}

#: Every calibration label, setting by setting.
CALIBRATION_LABELS = tuple(label for labels in SETTING_LABELS.values() for label in labels)

#: The SD of ``sqrt(n)`` times each calibration label's estimate at the limit rule
#: ``d0 = I(b(W1) > 0)``: the efficient influence curve SD of the stratum or marginal value of
#: ``d0`` by the quadrature of :func:`~tests.studies._learned_rule_strata_law.rule_moments`.
#: They size the noise controls only, at one unit each, and no verdict reads an efficiency
#: ratio (``calibration_efficiency_ratio=False``). ``test_learned_rule_strata_design``
#: recomputes them.
EFFICIENCY_BOUNDS = {
    "unweighted_s0": 0.408456,
    "unweighted_s1": 0.332716,
    "unweighted_s2": 0.278199,
    "unweighted_all": 0.243423,
    "weighted_s0": 0.429930,
    "weighted_s1": 0.356453,
    "weighted_s2": 0.314792,
    "weighted_all": 0.268495,
    "binary_steep_s0": 1.441609,
    "binary_steep_s1": 1.145316,
    "binary_steep_s2": 0.937757,
    "binary_steep_all": 0.660536,
    "binary_steep_weighted_s0": 1.528730,
    "binary_steep_weighted_s1": 1.222094,
    "binary_steep_weighted_s2": 1.039668,
    "binary_steep_weighted_all": 0.716352,
}

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The invalid-inference control of every stratum label: the stratum curve without its
#: ``1 / P_{n,v}(S = s)`` normalisation (E1's control, plan section 3).
UNNORMALISED = "unnormalised_curve_control"

#: The invalid-inference control of the marginal label of the bounded settings: the marginal
#: curve without its share term. Its asymptotic SE ratio at the limit rule is 0.778 rows only
#: and 0.773 weighted. No binary label carries one: the Bernoulli residual variance leaves the
#: ratio at 0.98, above the band floor 0.93, so it could not fail.
SHARE_DROPPED = "share_dropped_control"

#: The marginal labels whose share-term control is declared.
SHARE_DROPPED_LABELS = tuple(f"{setting}_all" for setting in law.BOUNDED_SETTINGS)

#: The labels of the stratum parameters, which carry :data:`UNNORMALISED`.
STRATUM_LABELS = tuple(label for label in CALIBRATION_LABELS if not label.endswith("_all"))

#: The targeting-control labels: every stratum parameter of the bounded rows-only setting.
NECESSITY_LABELS = tuple(
    label for label in SETTING_LABELS["unweighted"] if not label.endswith("_all")
)


def calibration_cells() -> tuple[str, ...]:
    """Every ``interval_calibration`` cell, in label order."""
    cells: list[str] = []
    for label in CALIBRATION_LABELS:
        cells.extend(
            f"{label}__{kind}" for kind in (POSITIVE, "shrunken_se_control", "noise_control")
        )
        if label in STRATUM_LABELS:
            cells.append(f"{label}__{UNNORMALISED}")
        if label in SHARE_DROPPED_LABELS:
            cells.append(f"{label}__{SHARE_DROPPED}")
    return tuple(cells)


PROPERTY_CELLS = {
    "interval_calibration": calibration_cells(),
    "targeting_necessity": tuple(
        f"{label}__{arm}" for label in NECESSITY_LABELS for arm in ("targeted", "untargeted")
    ),
}

#: The truth-harness artifact: one row per primary replication and parameter.
HARNESS_ARTIFACT = "strata-harness.csv.gz"

STUDY = StudyRecord(
    name="Learned-rule value inside baseline strata",
    slug="learned-rule-strata",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_strata",
    document="docs/technical-reference/method-evidence/learned-rule-strata.md",
    anchor="learned-rule-value-inside-baseline-strata",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-strata",
    reference=None,
    failure_bound={"cleverly-learned-rule-strata": 0.01},
    modules=(
        "tests/studies/learned_rule_strata.py",
        "tests/studies/learned_rule_strata_properties.py",
        "tests/studies/_learned_rule_strata_law.py",
        "tests/studies/_fold_evaluated_strata_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/_strata.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/learners/crossfit.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_strata",
    properties_module="tests.studies.learned_rule_strata_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(outcome_learner=<outcome learner>, treatment_learner=<treatment learner>, "
        "learned_rule=LearnedRule(), cross_fit=True, n_folds=5, cv_evaluation=True, "
        "targeting_scheme='pooled', g_bounds='auto', q_bounds=<(0.0, 1.0) on the beta "
        "outcome, None on the binary one>, simultaneous=False, "
        "random_state=<fold seed>).fit(frame, outcome='Y', treatment='A', "
        "covariates=['W1', 'W2', 'S'], strata=['S'], weights=<'w' in a weighted setting>); "
        "n = 1000, rows only, complete data, repeats=1"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "QuasiBinomialGLM(tol=1e-10)) on (A, W1, W2, S) for the beta outcome; the same "
        "products with LogisticRegression(C=1e6, max_iter=5000, tol=1e-10) for the binary one"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2, S)",
    "targeting_control_learner": (
        "E1's drop_w1_learner(): columns 0 and 2 of (A, W1, W2, S), so A and W2 only, and "
        "their product, quasibinomial; it learns a rule of W2 alone, and the truth is refit "
        "with it"
    ),
    "law": (
        "the law of tests/studies/_fold_evaluated_strata_law.py: W1 ~ U(-1, 1), "
        "W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, Y | A, W ~ Beta(12 m, 12 (1 - m)) "
        "with blip b = 0.1 + 1.5 W1 (unweighted, weighted; rule boundary W1 = -1/15 in "
        "stratum 1) or Bernoulli(m) with b = -1.2 + 3.0 W1 (binary_steep, "
        "binary_steep_weighted; boundary W1 = 0.4 in stratum 2), m = Qbar0(A, W), logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 "
        "+ a b; S = 0, 1, 2 on W1 in [-1, -0.6), [-0.6, 0.1), [0.1, 1] (shares 0.2, 0.35, "
        "0.45); weighted settings w = 1 + 1.5 W2"
    ),
    "truth": (
        "per replication: refit each fold's plug-in rule on the fold's training rows with the "
        "fit's outcome learner, task and scaled outcome; require it on every validation row "
        "(rule L4) and the stored fold rule on every row; integrate Qbar0(d_v(W), W) over each "
        "stratum's interval of W1 by the trapezoid rule on 4001 points for each W2, weighted by "
        "w(W2); each stratum truth is the mean over the 5 folds, and the marginal truth is the "
        "share mixture"
    ),
    "oracle_se": (
        "sqrt(V^-2 sum_v Var D_{s,v} / n_v) at each fold's rule, by the same quadrature; "
        "descriptive"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(scenario, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """One primary replication: one row per reported parameter, each with its own truth."""
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, scenario, fold_seed)
    rows = []
    for label in law.PARAMETER_LABELS:
        name = law.parameter(label)
        truth = fitted.truth.truths[name]
        estimate = fitted.result.estimates[name]
        std_error, low, high = reported_inference(estimate)
        rows.append(
            {
                "implementation": record.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": truth,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": law.untargeted(fitted.result, label),
                "oracle_se": fitted.truth.oracle_ses[name],
                "rule_rows_checked": fitted.truth.rows_checked,
                "solver_warnings": fitted.warnings,
            }
        )
    return rows


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.draw(scenario, size, replicate_seed(record, scenario, replicate))


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
        partial(primary_rows, record),
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
