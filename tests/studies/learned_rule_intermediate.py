r"""Registered evidence for the learned rule with an intermediate held at each level (X11 (j)).

The subject is ``TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto").fit(..., intermediate="Z")``, with every other
setting at the constructor's default. The fit reports one result per level ``z`` of the binary
intermediate. Each level learns its own fold rules, the plug-in
:math:`d^z_{nv}(w) = 1\{\bar Q_{nv}(1, z, w) > \bar Q_{nv}(0, z, w)\}`, and reports the
fold-evaluated value ``ey_learned_rule[learned rule]``, whose target is the data-adaptive
:math:`\tilde\psi^z_{0n} = V^{-1}\sum_v E_0\bar Q_0(d^z_{nv}(W), z, W)`. The clever covariate is
:math:`1\{A = d^z_i\}1\{Z = z\}/(g(A \mid W) q(z \mid A, W))`.

The study has one law (:mod:`tests.studies._learned_rule_intermediate_law`), RM30's with a
binary ``Z`` after ``A``, ``logit P(Z = 1 | A, W) = 0.2 + 0.6 A - 1.2 W1``, whose effect on ``Y``
interacts with ``A``: the blip is ``0.1 + W1`` at ``z = 0`` (RM30's ``non_exceptional``) and
``-0.5 + W1`` at ``z = 1``, so the optimal rules differ on 30% of the population. Every cell is at
n = 2,000. The truth of a learned-rule row is its level's value of the fit's own fold rules, by
refit (rule L4) and quadrature, so the record declares ``truth_varies_by_replicate``.

``cde_z0`` and ``cde_z1``
    The primary scenarios: the ``declared`` (correct) outcome learner, both levels of one fit,
    one sample per replication (``cde_z1`` draws ``cde_z0``'s sample). The primary gate reads
    each level's bias leg, coverage floor and SE sanity band.
``interval_calibration``
    ``learned_z0`` and ``learned_z1``, the declared fit at each level, with the SE x 0.8 and noise
    controls and the ``term_dropped_control``: the same estimate with the SE of the curve
    rebuilt without the factor :math:`1\{Z = z\}/q`, which must fall below the SE-ratio band
    (witness W-j3's mutation, an invalid-inference control). ``known_z0`` and ``known_z1``, the
    known rule ``1{W1 > 0}`` through the same fold-evaluated controlled-direct-effect path, with
    the SE x 0.8 and noise controls: the first registered evidence for that path, which witness
    W-j1 reduces the learned fit to bit for bit, so W-j1 becomes a reduction to a measured path.
``targeting_necessity``
    The ``drop_w1`` learner at each level: the targeted value against the untargeted fold
    average of the initial plug-ins, whose limit is biased by the confounding through ``W1``
    (0.098 at ``z = 0``, -0.025 at ``z = 1``, by quadrature).

The X11 wave-2 plan (``X11-wave2-plan.md`` rev 2.1 in the plans repository, section 7), the
implementation notes and review ``reviews/x11/X11-J-impl-notes.md`` and
``reviews/x11/X11-J-impl-review.md`` (finding 6: the known-rule cell is a must-have), and the
study design record ``reviews/x11/X11-J-study-design.md`` declare every quantity here before any
run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_intermediate_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The primary replications: the ``declared`` fit at both levels. PRIMARY_REPLICATES is the
#: smallest multiple of 400 of at least 1,200 whose predictive probability that both levels'
#: bias legs and coverage floors pass is at least 0.95, on the 4,000-draw pilot and on the 8,000
#: draws pooled with the probe (``test_learned_rule_intermediate_design``).
PRIMARY_REPLICATES = 1_200
PRIMARY_N = 2_000
#: The record seeds: the first candidate from 20270151 in steps of 10 whose declared sample seeds
#: meet no other X11 or RM30 study's and no pilot seed, checked on the seeds alone.
SEED = 20271311
RESAMPLING_SEED = 20271312

#: The primary scenario of each level. ``cde_z1`` draws ``cde_z0``'s sample, so one fit writes
#: both levels' rows.
LEVEL_SCENARIOS = {z: f"cde_z{int(z)}" for z in law.LEVELS}
PRIMARY_SCENARIO = LEVEL_SCENARIOS[0.0]
SCENARIOS = dict.fromkeys(LEVEL_SCENARIOS.values(), (law.ESTIMAND,))
SCENARIO_LEVELS = {scenario: z for z, scenario in LEVEL_SCENARIOS.items()}

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The calibration labels: the learned rule and the known rule at each level.
LEARNED_LABELS = tuple(f"learned_z{int(z)}" for z in law.LEVELS)
KNOWN_LABELS = tuple(f"known_z{int(z)}" for z in law.LEVELS)
CALIBRATION_LABELS = (*LEARNED_LABELS, *KNOWN_LABELS)

#: The SD of ``sqrt(n)`` times each label's estimate at its limit rule, by quadrature
#: (:func:`~tests.studies._learned_rule_intermediate_law.level_limit`): the optimal rule of the
#: level for the learned labels (the declared learner is correct) and ``1{W1 > 0}`` for the known
#: ones. They size the noise controls only, at one unit each, and no verdict reads an efficiency
#: ratio. ``test_learned_rule_intermediate_design`` recomputes them.
EFFICIENCY_BOUNDS = {
    "learned_z0": 1.120917,
    "learned_z1": 0.882374,
    "known_z0": 1.105274,
    "known_z1": 0.855360,
}

#: The invalid-inference control of each learned label: the same estimate, with the SE of the
#: curve without ``1{Z = z} / q``. It must lie wholly below the SE-ratio band.
TERM_DROPPED = "term_dropped_control"

#: The bias control (plan section 7): the ``drop_w1`` learner at each level. The positive arm is
#: the targeted value; the control is the same fit's untargeted fold average of the initial
#: plug-ins.
NECESSITY_FAMILY = "targeting_necessity"
NECESSITY_LABELS = tuple(f"drop_w1_z{int(z)}" for z in law.LEVELS)
NECESSITY_ARMS = ("targeted", "untargeted")

PROPERTY_CELLS = {
    "interval_calibration": (
        *(
            f"{label}__{kind}"
            for label in LEARNED_LABELS
            for kind in (POSITIVE, "shrunken_se_control", "noise_control", TERM_DROPPED)
        ),
        *(
            f"{label}__{kind}"
            for label in KNOWN_LABELS
            for kind in (POSITIVE, "shrunken_se_control", "noise_control")
        ),
    ),
    NECESSITY_FAMILY: tuple(
        f"{label}__{arm}" for label in NECESSITY_LABELS for arm in NECESSITY_ARMS
    ),
}

#: The support rows the registering change adds, declared before any run.
#:
#: ``learned``: the P11 row of the fold-evaluated learned rule with ``missing=("intermediate",)``.
#: Added if, and only if, these verdicts pass as declared: the primary performance tests of both
#: scenarios (bias leg, coverage floor, SE sanity); the ``learned_z0`` and ``learned_z1``
#: calibration verdicts, positive and all three controls; and the ``targeting_necessity``
#: verdicts.
#:
#: ``known``: the P5 row of a known regime through the fold-evaluated controlled-direct-effect
#: path, ``missing=("intermediate",)``. Added if, and only if, the ``known_z0`` and ``known_z1``
#: calibration verdicts pass, positive and both controls.
#:
#: Neither row covers weights, clusters, strata, fold targeting, repeats, the stacked report,
#: the in-sample fit, more than two arms, known regimes beside the learned rule, missing
#: outcomes or a rule learner. A red cell is diagnosed before any route is chosen; with no
#: defect found it is published red under ``reporting`` with an owner row, and its row is not
#: added.
SUPPORT_ROWS = {
    "learned": {
        "splitting": ("fold_evaluated",),
        "families": ("P11",),
        "missing": ("intermediate",),
    },
    "known": {
        "splitting": ("fold_evaluated",),
        "families": ("P5",),
        "missing": ("intermediate",),
    },
}

#: The truth-harness artifact: one row per primary replication and level.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned treatment rules with an intermediate held at each level",
    slug="learned-rule-intermediate",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_intermediate",
    document="docs/technical-reference/method-evidence/learned-rule-intermediate.md",
    anchor="learned-treatment-rules-with-an-intermediate-held-at-each-level",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    scenario_seed_owners={LEVEL_SCENARIOS[1.0]: PRIMARY_SCENARIO},
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-intermediate",
    reference=None,
    failure_bound={"cleverly-learned-rule-intermediate": 0.01},
    modules=(
        "tests/studies/learned_rule_intermediate.py",
        "tests/studies/learned_rule_intermediate_properties.py",
        "tests/studies/_learned_rule_intermediate_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/estimators/direct_effect.py",
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
    runner_module="tests.studies.learned_rule_intermediate",
    properties_module="tests.studies.learned_rule_intermediate_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, "
        "targeting_scheme='pooled', g_bounds='auto', random_state=<fold seed>).fit(..., "
        "intermediate='Z') at n = 2000; every other setting the constructor's default; repeats=1"
    ),
    "known_rule": (
        "the same estimator with learned_rule=None and interventions=[Rule(1{W1 > 0}, "
        "rule_kind='known')]: the fold-evaluated controlled-direct-effect path"
    ),
    "rule": "1{Qbar_v(1, z, w) > Qbar_v(0, z, w)} of each fold's outcome model at each level z; "
    "a tie to arm 0",
    "outcome_learners": {
        "declared": (
            "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
            "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2, Z)"
        ),
        "drop_w1": (
            "the columns (A, W2, Z), PolynomialFeatures(3, interaction_only=True, "
            "include_bias=False), LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)"
        ),
    },
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "intermediate_learner": "LogisticRegression(C=1e6, max_iter=1000) on (A, W1, W2)",
    "law": (
        "W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0(1 | W) = 0.3 W1 - 0.2 W2, logit P(Z = 1 | "
        "A, W) = 0.2 + 0.6 A - 1.2 W1, logit Qbar0(a, z, W) = 0.2 + 0.5 W1 - 0.3 W2 + 0.3 z + a "
        "(0.1 + W1 - 0.6 z)"
    ),
    "cells": {
        "learned": (
            "the primary (cde_z0, cde_z1) and interval_calibration learned_z0, learned_z1 (SE x "
            "0.80, noise and term_dropped controls)"
        ),
        "known": "interval_calibration known_z0, known_z1 (SE x 0.80 and noise controls)",
        "drop_w1": "targeting_necessity: targeted value against the untargeted plug-in",
    },
    "variance": "V^-2 sum_v n_v^-2 sum_{i in v} D_i^2 of the stored curve",
    "term_dropped": (
        "the same variance of the curve with H = 1{A = d} / g, without 1{Z = z} / q; the "
        "harness also requires the curve with the factor to give the reported variance within "
        "a relative 1e-8"
    ),
    "truth": (
        "per replication and level: refit each fold's outcome learner on the fit's training rows "
        "with the package's fit_on_rows, take its rule at z, require it to equal the fit's rule "
        "on every validation row and the stored fold rule of the level on every row (rule L4), "
        "and integrate Qbar0(d_v(W), z, W) by the trapezoid rule on 4001 points of W1 for each "
        "W2; weight 1/V. The known rule's truth is its value at z by the same quadrature"
    ),
    "oracle_se": "sqrt(V^-2 sum_v Var D*_z(d_v) / n_v) at the fold rules; descriptive",
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', 'cde_z0', replicate)",
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
    record: StudyRecord, key: tuple[tuple[str, ...], int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    _, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", PRIMARY_SCENARIO, replicate)
    fitted = law.measured(frame, "declared", fold_seed)
    rows = []
    for z, scenario in LEVEL_SCENARIOS.items():
        estimate = fitted.estimate(z)
        std_error, low, high = reported_inference(estimate)
        truth = fitted.truths[z].truth
        rows.append(
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
                "initial_estimate": law.untargeted_estimate(fitted.results[z]),
                **law.harness_columns(fitted, z),
            }
        )
    return rows


def _sample(record: StudyRecord, key: tuple[tuple[str, ...], int, int]) -> pd.DataFrame:
    _, replicate, size = key
    return law.draw(size, replicate_seed(record, PRIMARY_SCENARIO, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``, both levels from one fit.

    ``record`` is the declared one, or the throwaway record of a smoke run (rule L9).
    """
    scenarios = tuple(record.scenarios)
    outcomes = replication.fit_primary(
        [(scenarios, replicate, n) for replicate in range(replicates)],
        partial(_primary, record),
        draw=partial(_sample, record),
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, PRIMARY_SCENARIO, key[1]),
        cell_of=lambda key: ";".join(key[0]),
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *law.HARNESS_COLUMNS]])


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication and level."""
    columns = ["scenario", "replicate", "truth", *law.HARNESS_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "replicate"], ignore_index=True
        )
    }
