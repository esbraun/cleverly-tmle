r"""Registered evidence for the learned two-node regimen of ``LTMLE`` (X11 (b)).

The subject is ``LTMLE(LearnedRegimen(), n_folds=10, random_state=<fold seed>)`` with the
study's learners and every other setting at the constructor's default, on a censored two-node panel with a binary
treatment at each node and an end-of-study binary outcome. Each training fold ``v`` learns the
rule pair :math:`d_{nv}` by backward plug-in Q-learning inside its own untargeted recursion, each
validation row is evaluated under its own fold's pair, and the fit reports the stacked value
``ey_regimen[learned regimen]``, whose target is the data-adaptive
:math:`\tilde\psi_{0n} = \sum_v (n_v / n) \Psi_{d_{nv}}(P_0)`. The variance is the stacked curve
centred at the pooled estimate.

The study has one law (:mod:`tests.studies._learned_regimen_law`): ``L2`` is lowered by 0.8 under
``A1 = 1``, the optimal node-2 rule treats 34% and the optimal node-1 rule 36%, and each boundary
has a margin (the blip's slope is 1.2 at node 2; the node-1 value gap has a nonzero slope in
``W1``). The optimal value is 0.557923 and the best static regimen's 0.4600. Every cell is at
n = 2,000. The truth of a row is the value of the fit's own fold rule pairs, by refit (rule L4)
and quadrature over ``L2``, so the record declares ``truth_varies_by_replicate``.

``two_node``
    The primary scenario: the ``declared`` learners, one fit per replication. The primary gate
    reads the bias leg, the coverage floor and the SE sanity band.
``interval_calibration``
    ``two_node``, the declared fit, with the SE x 0.8 and noise controls.
``conservative_coverage``
    ``squared_w1``: the node-1 learner on ``(W1^2, W2)``, which keeps the confounder ``W1`` of
    ``A1`` only through its square, with the treatment and censoring models correct. Its limit
    rule treats where ``|W1|`` is large (50% at node 1), is worth 0.5046, 0.053 below the
    optimum, and crosses its boundary with a nonzero slope, so it has a margin. The targeted
    value is held to the bias leg and a coverage floor of 0.92, with no SE-ratio gate. The
    control is the same estimate with its SE x 0.8, whose coverage must lie wholly below the
    floor.
``targeting_necessity``
    ``drop_w1``: the node-1 learner on ``W2`` alone. The targeted value against the untargeted
    plug-in, the sample mean of the stitched node-1 initial values. The plug-in's limit misses its
    own rule pair's value by 0.0596, 2.65 SEs at n = 2,000, by quadrature. The learner's limit
    arm contrast is only 0.015 to 0.027, so its rules flip between folds and its interval
    under-covers; no verdict of this family reads coverage (the design record says why the
    learner moved out of ``conservative_coverage``).
``fold_locality``
    The declared learners on one sample and fold seed twice: the fit (``fold_local``) against the
    same fit with each fold's rules learned on its validation rows (``validation_rows``, RM30's
    ``ValidationRowRule`` pattern), each read against its own rule pairs' value. The control's
    optimism must be resolved.

The X11 wave-2 plan (``X11-wave2-plan.md`` rev 2.1 in the plans repository, section 8), the
implementation review ``reviews/x11/X11-K-impl-review.md`` (note 8) and the study design record
``reviews/x11/X11-K-study-design.md`` declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_regimen_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The primary replications of the ``declared`` fit: the smallest multiple of 400 of at least
#: 1,200 whose predictive probability that the bias leg and the coverage floor pass is at least
#: 0.95 on the 4,000-draw pilot (``test_learned_regimen_ltmle_design``).
PRIMARY_REPLICATES = 1_200
PRIMARY_N = 2_000
#: The record seeds: the first candidate from 20271421 in steps of 10 whose declared sample seeds
#: meet no other X11 or RM30 study's and no pilot seed, checked on the seeds alone. 20271421
#: drew one sample seed of another X11 study.
SEED = 20271431
RESAMPLING_SEED = 20271432

PRIMARY_SCENARIO = law.LAW
SCENARIOS = {PRIMARY_SCENARIO: (law.ESTIMAND,)}

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The calibration label: the declared fit.
CALIBRATION_LABELS = (law.LAW,)

#: The SD of ``sqrt(n)`` times the estimate at the optimal rule pair, by quadrature
#: (:func:`~tests.studies._learned_regimen_law.law_limit`). It sizes the noise control only, at
#: one unit, and no verdict reads an efficiency ratio. ``test_learned_regimen_ltmle_design``
#: recomputes it.
EFFICIENCY_BOUNDS = {law.LAW: 0.894934}

#: The misspecified node-1 learners (plan section 8: "a misspecified node-1 learner with g
#: correct"). ``conservative_coverage`` reads ``squared_w1``, least squares on ``(W1^2, W2)``;
#: ``targeting_necessity`` reads ``drop_w1``, least squares on ``W2`` alone.
COVERAGE_FAMILY = "conservative_coverage"
COVERAGE_LABEL = "squared_w1"
#: The ``conservative_coverage`` control: the same estimate with its SE times this factor.
COVERAGE_CONTROL = "shrunken_coverage_control"
COVERAGE_SHRINK = 0.80

NECESSITY_FAMILY = "targeting_necessity"
NECESSITY_LABEL = "drop_w1"
NECESSITY_ARMS = ("targeted", "untargeted")

LOCALITY_FAMILY = "fold_locality"
LOCALITY_LABEL = law.LAW
LOCALITY_ARMS = ("fold_local", "validation_rows")

PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{law.LAW}__{kind}" for kind in (POSITIVE, "shrunken_se_control", "noise_control")
    ),
    COVERAGE_FAMILY: tuple(f"{COVERAGE_LABEL}__{kind}" for kind in (POSITIVE, COVERAGE_CONTROL)),
    NECESSITY_FAMILY: tuple(f"{NECESSITY_LABEL}__{arm}" for arm in NECESSITY_ARMS),
    LOCALITY_FAMILY: tuple(f"{LOCALITY_LABEL}__{arm}" for arm in LOCALITY_ARMS),
}

#: The support row the registering change adds, declared before any run.
#:
#: ``learned``: row ``G47``, ``long_row("G47", ["learned-regimen-ltmle"], regimens=("learned",),
#: splitting=("cross_fitted",))``, the key of every fit of this study: ``LTMLE``, an end-of-study
#: binary outcome at two binary nodes, censoring, cross-fitted, no weights, no clusters, no known
#: factor and every option at its default (``test_learned_regimen_ltmle_design`` builds it and
#: compares it with a fit's key). It licenses ``point`` and ``wald``. It is added if, and only if,
#: these verdicts pass as declared: the primary performance tests (bias leg, coverage floor, SE
#: sanity); the ``interval_calibration`` verdicts, the positive and both controls; the
#: ``conservative_coverage`` verdicts, the positive and its control; the ``targeting_necessity``
#: verdicts; and the ``fold_locality`` verdicts.
#:
#: The row would not cover a continuous outcome with ``q_bounds=``, declared node bounds,
#: ``ratio_trim=``, a stopping rule or a known mechanism (admitted experimental at fit and keyed
#: apart), nor a misspecified treatment or censoring model: no cell measures them. Every refused
#: composition of the implementation (more than two nodes, ``repeats > 1``, ``n_folds=1``,
#: ``weights=``, ``id=``, a selector as the node-1 regression, ...) stays refused. A red cell is
#: diagnosed before any route is chosen; with no defect found it is published red under
#: ``reporting`` with an owner row, and the row is not added.
SUPPORT_ROWS = {
    "learned": {
        "row": "G47",
        "regimens": ("learned",),
        "splitting": ("cross_fitted",),
        "missing": ("censoring",),
        "kinds": ("point", "wald"),
    },
}

#: The truth-harness artifact: one row per primary replication.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned two-node regimens",
    slug="learned-regimen-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "learned_regimen_ltmle",
    document="docs/technical-reference/method-evidence/learned-regimen-ltmle.md",
    anchor="learned-two-node-regimens",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-regimen-ltmle",
    reference=None,
    failure_bound={"cleverly-learned-regimen-ltmle": 0.01},
    modules=(
        "tests/studies/learned_regimen_ltmle.py",
        "tests/studies/learned_regimen_ltmle_properties.py",
        "tests/studies/_learned_regimen_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/longitudinal/estimator.py",
        "src/cleverly/longitudinal/learned.py",
        "src/cleverly/longitudinal/sequential.py",
        "src/cleverly/longitudinal/regimen.py",
        "src/cleverly/interventions/base.py",
        "src/cleverly/_support.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_regimen_ltmle",
    properties_module="tests.studies.learned_regimen_ltmle_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "LTMLE(LearnedRegimen(), n_folds=10, random_state=<fold seed>, outcome_learner=<node "
        "2>, pseudo_learner=<node 1>, treatment_learner=<logistic>, censoring_learner="
        "<logistic>).fit(panel) at n = 2000; every other setting the constructor's default; "
        "repeats=1"
    ),
    "rule": (
        "per training fold, backward plug-in Q-learning: d2(a1, h) = 1{Qbar2(a1, 1, h) > "
        "Qbar2(a1, 0, h)} per first-node arm, then d1(w) = 1{Q1(1, w) > Q1(0, w)} of the "
        "node-1 regression of Qbar2 at the branch rule; a tie to arm 0"
    ),
    "learners": {
        "node_2": "LogisticRegression(C=1e6, max_iter=1000) per stratum (A1, A2) on (W1, W2, L2)",
        "declared_node_1": (
            "Pipeline(PolynomialFeatures(3), LinearRegression()) on (W1, W2) per first-node arm"
        ),
        "squared_w1_node_1": (
            "ColumnTransformer(FunctionTransformer(np.square) on W1, passthrough W2) then "
            "LinearRegression() per first-node arm"
        ),
        "drop_w1_node_1": (
            "ColumnTransformer(passthrough W2) then LinearRegression() per first-node arm"
        ),
        "treatment_and_censoring": "LogisticRegression(C=1e6, max_iter=1000), main terms",
    },
    "law": (
        "W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5); logit g1 = 0.8 W1 - 0.3 W2; logit P(C1 = 1) = 2.5 "
        "+ 0.3 W1; L2 ~ N(0.5 W1 + 0.2 W2 - 0.8 A1, 1); logit g2 = -0.2 + 0.3 L2 + 0.4 A1; "
        "logit P(C2 = 1) = 2.5 - 0.2 L2; logit Qbar0 = -0.2 + 0.6 W1 - 0.2 W2 + 0.3 L2 + a2 "
        "(-0.1 + 1.2 L2 - 0.6 a1) + a1 (0.3 + 1.2 W1)"
    ),
    "cells": {
        "declared": (
            "the primary (two_node), interval_calibration two_node (SE x 0.80 and noise "
            "controls) and fold_locality (the fold-local fit against rules learned on each "
            "fold's validation rows)"
        ),
        "squared_w1": "conservative_coverage (SE x 0.80 coverage control)",
        "drop_w1": "targeting_necessity (targeted value against the untargeted plug-in)",
    },
    "variance": "sum_i (Z_i - psi)^2 / (n (n - 1)) of the stacked curve centred at the estimate",
    "truth": (
        "per replication: refit each fold's backward Q-learning on the fit's training rows with "
        "scikit-learn alone, require both rules to equal the stored fold rules on every row "
        "(rule L4), and integrate each fold's pair by the trapezoid rule on 1001 points of W1 "
        "for each W2 and Gauss-Legendre (24 nodes each side of the node-2 threshold) over L2; "
        "weight n_v / n. The validation_rows arm's truth is the value of its own rule pairs"
    ),
    "exactness": (
        "the harness rebuilds the stacked curve from the stored node steps and requires its "
        "variance to equal the reported one within a relative 1e-8"
    ),
    "oracle_se": "sqrt(sum_v n_v Var D*(d_v)) / n at the fold rule pairs; descriptive",
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', 'two_node', replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, "declared", fold_seed)
    estimate = fitted.estimate
    std_error, low, high = reported_inference(estimate)
    truth = fitted.truth.truth
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
            "initial_estimate": law.untargeted_estimate(fitted.result),
            **law.harness_columns(fitted),
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.draw(size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``.

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
        cell_of=lambda key: key[0],
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
