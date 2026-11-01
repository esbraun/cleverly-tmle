r"""Registered evidence for the stacked report of the learned-rule value (X11 PR F1, part (e)).

The subject is ``TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=False,
targeting_scheme="pooled", g_bounds="auto")``: one pooled fluctuation over every validation row
with the observation weights, the estimate :math:`\hat\psi = \sum_v (W_v/W)\psi^*_{nv}`, the
curve :math:`\phi_i = w_i(Z_i - \hat\psi)`, and the stored variance the centred rule
:math:`\{n(n-1)\}^{-1}\sum_i(\phi_i - \bar\phi)^2` (rows and weights) or the cluster-total rule
:math:`J\,\widehat{\mathrm{var}}(T_j)/n^2` with :math:`\widehat{\mathrm{var}}` of divisor
:math:`J - 1` (``id=``, normal reference at 200 clusters). Three settings:

``rows``
    The RM30 ``non_exceptional`` law at n = 2,000; the ten folds are equal, so
    :math:`W_v/W = 1/V` and the cell measures the pooled centring.
``weighted``
    The ``learned-rule-compositions`` weighted law (``misspecified_limit`` blip, known weight
    ``1 + 1.5 * 1{W1 > 0}``) at n = 2,000; the fold masses differ.
``clustered``
    200 clusters of size 5 to 15 that increases with the cluster's outcome effect, about 2,000
    rows (:mod:`tests.studies._learned_rule_stacked_law`). Its fold-evaluated fit on the same
    draws is the reporting row ``fold_evaluated_reporting/clustered__fold_evaluated``.

The target of each replication is data-adaptive, :math:`\sum_v (W_v/W)\Psi_{d_{nv}}(P_0)`, so
each row carries its own truth. The X11 wave-2 plan (``X11-wave2-plan.md`` rev 2.1 in the plans
repository, section 2), its review, the implementation review ``reviews/x11/X11-F1-impl-review.md``
and the study design record ``reviews/x11/X11-F1-study-design.md`` declare every quantity here
before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_stacked_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 2_000
#: The nominal row count. A clustered draw holds ``PRIMARY_N // 10`` clusters of mean size 10
#: and publishes this nominal ``n``; its ``rows`` column holds the realized count.
PRIMARY_N = 2_000
SEED = 20263071
RESAMPLING_SEED = 20263072

SCENARIOS = dict.fromkeys(law.SCENARIOS, (law.ESTIMAND,))

#: The SD of ``sqrt(n)`` times the estimate at each scenario's limit rule, the rule of the
#: declared learner's population projection (:func:`~tests.studies._learned_rule_stacked_law.
#: efficiency_bound`). ``rows`` and ``weighted`` equal the ``learned-rule-compositions`` values of
#: the same laws (``fold_targeted`` and ``weighted``); ``clustered`` is ``sqrt(E(T^2) / E(N))``
#: under the size law. They size the noise controls only, at one unit each, and no verdict reads
#: an efficiency ratio. ``test_learned_rule_stacked_design`` recomputes them.
EFFICIENCY_BOUNDS = {
    "rows": 0.664438,
    "weighted": 0.715236,
    "clustered": 0.852277,
}

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The clustered cell's own control: the stacked estimate with the row-level centred SE of the
#: same curve, which ignores the clusters. At the limit rule its SE over the declared SE is
#: ``sqrt(row variance / (E(T^2) / E(N)))`` = 0.771, so it must fall below the band.
ROW_VARIANCE = "row_variance_control"

#: The reporting family: the fold-evaluated fit of every clustered calibration draw, with the
#: role ``diagnostic``. No margin is read and no verdict forms.
REPORTING_FAMILY = "fold_evaluated_reporting"
REPORTING_CELL = "clustered__fold_evaluated"

PROPERTY_CELLS = {
    "interval_calibration": (
        *tuple(
            f"{label}__{kind}"
            for label in law.SCENARIOS
            for kind in (POSITIVE, "shrunken_se_control", "noise_control")
        ),
        f"clustered__{ROW_VARIANCE}",
    ),
    REPORTING_FAMILY: (REPORTING_CELL,),
}

#: The rule, declared before any run, that decides the ``unequal_clusters`` qualifier of a
#: stacked learned-rule fit (plan section 2, "_support.py"; plan review finding 3). The
#: registering change drops the qualifier from fits keyed ``stacked`` if and only if every
#: clause holds on the committed artifact; otherwise it stays, and the registered clustered row
#: names it. The fold-evaluated report keeps the qualifier either way.
UNEQUAL_CLUSTERS_RULE = (
    "clustered__declared passes the interval_calibration rule (coverage and SE-ratio bands)",
    "clustered__declared establishes its bias inside the margin (bias_equivalent)",
    "clustered__row_variance_control, clustered__shrunken_se_control and "
    "clustered__noise_control each fail as declared",
    "no clustered replication was lost beyond the failure bound",
)

#: What a dropped qualifier would claim, and no more: stacked learned-rule fits with ``id=`` at
#: about 200 clusters in 10 folds, sizes 5 to 15 that rise with the outcome effect. The
#: registering change states this scope on the support row.
UNEQUAL_CLUSTERS_SCOPE = (
    "200 clusters of size 5 to 15 in 10 folds, the size rising with the cluster's outcome "
    "effect (size-curve correlation 0.891, curve ICC 0.077)"
)

#: The truth-harness artifact: one row per primary replication.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Stacked report of the learned-rule value",
    slug="learned-rule-stacked",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_stacked",
    document="docs/technical-reference/method-evidence/learned-rule-stacked.md",
    anchor="stacked-report-of-the-learned-rule-value",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-stacked",
    reference=None,
    failure_bound={"cleverly-learned-rule-stacked": 0.01},
    modules=(
        "tests/studies/learned_rule_stacked.py",
        "tests/studies/learned_rule_stacked_properties.py",
        "tests/studies/_learned_rule_stacked_law.py",
        "tests/studies/_learned_rule_compositions_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/inference/cluster.py",
        "src/cleverly/learners/crossfit.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_stacked",
    properties_module="tests.studies.learned_rule_stacked_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=False, "
        "targeting_scheme='pooled', g_bounds='auto', random_state=<fold seed>) at nominal "
        "n = 2000; weights='w' (weighted); id='cluster' (clustered); repeats=1"
    ),
    "reporting_fit": "the same with cv_evaluation=True on every clustered calibration draw",
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "laws": {
        "rows": "RM30 non_exceptional, drawn by RM30's draw",
        "weighted": (
            "RM30 misspecified_limit, weight w = 1 + 1.5 * 1{W1 > 0}, drawn by the "
            "learned-rule-compositions draw"
        ),
        "clustered": (
            "RM30 non_exceptional logit plus 1.0 z_j, z_j ~ N(0, 1) per cluster; size "
            "N_j = 5 + Binomial(10, expit(1.5 z_j)); 200 clusters; Qbar0 the size-biased "
            "marginal over z by 64-node Gauss-Hermite quadrature"
        ),
    },
    "variance": {
        "rows": "{n (n - 1)}^-1 sum_i (phi_i - phibar)^2, phibar = 0 to solver tolerance",
        "weighted": "the same rule of phi_i = w_i (Z_i - psi), mean-one weights",
        "clustered": "J var(T_j; ddof=1) / n^2 of the cluster totals T_j, normal reference",
    },
    "truth": (
        "per replication: refit each fold's outcome learner on the fit's own training rows "
        "with its weights and clusters, require the fit's rule on every validation row, "
        "integrate each fold rule's value by the trapezoid rule on 4001 points of W1 for each "
        "W2 (and the Hermite rule in z for clustered), and weight fold v by W_v / W from the "
        "fit's weights and folds, checked against the record's fold_weights; the reporting "
        "fit's truth weights the same values by 1 / V"
    ),
    "oracle_se": "sqrt(sum_v omega_v^2 Var D_v / n_v), row-level; descriptive",
    "variance_terms": (
        "cross_term and between_term: the two extra terms of the stacked variance identity "
        "over the fold-evaluated second moment; descriptive"
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


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    result, truth, solver_warnings = law.measured(
        frame, scenario, base.outcome_learner(), fold_seed
    )
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
            "initial_estimate": untargeted_estimate(result),
            **law.diagnostic_columns(result, truth, solver_warnings),
        }
    ]


def untargeted_estimate(result: Any) -> float:
    """The fit without its fluctuation: the fold-mass-weighted initial plug-in at its rule."""
    nuisance = result.nuisance
    control, treated = nuisance.outcome.levels
    scaler = nuisance.scaler
    q0 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[control]), dtype=float)
    q1 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[treated]), dtype=float)
    plug_in = np.where(base.rule_of(result), q1, q0)
    value = float(np.average(plug_in, weights=np.asarray(result.data.weights, dtype=float)))
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate (rule L4): {value}")
    return value


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
