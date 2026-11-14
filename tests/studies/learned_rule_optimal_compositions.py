r"""Registered evidence for the compositions of the value of the optimal rule (X11 (f3)).

The subject is ``TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"], n_folds=10,
cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`` with RM30's interaction
logistic outcome learner and the known treatment mechanism, under the five compositions that X11
PR 2 admits (:mod:`tests.studies._learned_rule_optimal_compositions_law`):

``weighted``
    ``weights="w"`` with ``w = 1 + A Y`` on RM30's ``non_exceptional`` law, and the weighted
    population's own mechanism :math:`g_w = g_0 E[w \mid A, W]/E[w \mid W]` passed as
    ``treatment_probabilities=``. The weight moves the optimal rule to
    :math:`d^w_0 = 1\{W_1 > -0.1 - \log 2\}`, so the target is
    :math:`\Psi^w_{d^w_0}(P_w) = 0.690108`, 0.0277 above the weighted value of the unweighted
    rule (0.662385) and 0.1130 above :math:`\Psi_{d_0}(P_0)`.
``strata``
    ``strata=["W2"]``: the two stratum optimal values (0.610609 and 0.543621) and the marginal
    one (0.577115).
``known_regimes``
    Beside ``Static(1)`` and ``Static(0)``: the contrasts ``ate_optimal_rule[...]``,
    :math:`\Psi_{d_0} - \Psi_a` (0.045499 and 0.064936).
``in_sample``
    ``cross_fit=False`` with RM30's learner marked parametric (condition C5a): 0.577115.
``intermediate``
    The intermediate law of ``learned-rule-intermediate`` at each level ``z``:
    :math:`\Psi^z_{d^z_0}` (0.577115 and 0.597071).

Every truth is a fixed parameter by adaptive quadrature split at the rule's threshold
(:func:`~tests.studies._learned_rule_optimal_compositions_law.label_limits`; the design test
recomputes each with a second, independent quadrature), so the record reads one truth per
label.
The estimator of each composition is the learned-rule fit with the same setting (W-f3-id), so
its error splits into the learned-rule claim with that setting and minus the regret; each row
carries the value of the fit's own fold rules (refit, rule L4) and the regret, descriptive.

Two blind spots rest on witnesses rather than cells. The intermediate cell cannot see a fit
that ignores ``Z``: the total-effect optimum (0.588) is 0.44 and 0.46 SEs from the two level
truths, and the level-blind rule 0.27 and 0.22 SEs, so neither would move a verdict. That rests
on W-f3-id (each composition is the learned-rule fit with the same setting, bit for bit, which
``learned-rule-intermediate`` measures level by level), on witness W-f3z (the level optimum
beats the total-effect rule by a value gap) and on the harness, which stops a draw whose level
rules differ from the refit's at that level (rule L4).
The stratum cells carry the ``unnormalised_curve_control`` of ``learned-rule-strata``, which
resolves a gross normalisation error only; the fold-local share itself rests on W-f3-id and on
the witnesses of ``test_learned_rule_strata``.

The X11 plan rev 2 (``rows/X11/plan.md`` in the plans repository, section 5), its review and the
PR 2 implementation notes declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_optimal_compositions_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

base = law.base

#: The primary replications of each setting, every label of the setting from one fit. The
#: primary gate reads each parameter's bias leg (the 99% interval of the mean error inside 0.25
#: SDs), coverage floor and SE sanity band against its fixed truth. 4,400 is the smallest
#: multiple of 400, at least the plan's 1,000, at which the nine parameters pass together with
#: predictive probability at least 0.95 on the sizing pilot (0.960; 0.949 at 4,000): the bias
#: legs bind, because the regret biases each estimate by up to 0.160 SDs against its fixed truth
#: (the weighted label, re-piloted under ``1 + A Y``; ``test_learned_rule_optimal_compositions_
#: design``).
PRIMARY_REPLICATES = 4_400
PRIMARY_N = 2_000
#: Chosen before any declared draw so that every declared sample seed is distinct, within the
#: study and from every other X11 study's and every committed X11 pilot's. ``RESAMPLING_SEED``
#: (the property draws) was first searched in steps of 10 from 20271122 (20271122 shared eight
#: seeds with other X11 studies, among them an RM30 ``non_exceptional`` sample at n = 2,000 row
#: for row with ``learned-rule-optimal-value``), and ``SEED`` (the primary draws) then moved one
#: step from 20277911, whose primary drew one seed of the ``learned-rule-ep`` pilot. When the
#: design review's counts (4,400 primary, 10,000 calibration) were declared, 20277912 drew two
#: seeds twice across settings (rule L3 compares one setting's draws only), so
#: ``RESAMPLING_SEED`` moved on in steps of 10 to the first clear value, and ``SEED`` stayed.
SEED = 20277921
RESAMPLING_SEED = 20283472

#: One scenario per setting, with the parameters it reports.
SCENARIOS = {setting: tuple(law.PARAMETERS[setting].values()) for setting in law.SETTINGS}

#: The truth and the SD of ``sqrt(n)`` times the estimate of each label, by quadrature
#: (:func:`~tests.studies._learned_rule_optimal_compositions_law.label_limits`;
#: ``test_learned_rule_optimal_compositions_design`` recomputes each). ``bound`` sizes the noise
#: controls only, at one unit, and no verdict reads an efficiency ratio.
LIMITS: dict[str, dict[str, float]] = {
    "weighted": {"truth": 0.690108, "bound": 0.604640},
    "strata_all": {"truth": 0.577115, "bound": 0.664444},
    "strata_s0": {"truth": 0.610609, "bound": 0.925806},
    "strata_s1": {"truth": 0.543621, "bound": 0.950968},
    "contrast_1": {"truth": 0.045499, "bound": 0.650760},
    "contrast_0": {"truth": 0.064936, "bound": 0.708934},
    "in_sample": {"truth": 0.577115, "bound": 0.664444},
    "intermediate_z0": {"truth": 0.577115, "bound": 1.120875},
    "intermediate_z1": {"truth": 0.597071, "bound": 0.882349},
}

#: The weighted value of the unweighted optimal rule, :math:`\Psi^w_{d_0}(P_w)`, where a fit
#: that learned its rule without the weights would land
#: (:func:`~tests.studies._learned_rule_optimal_compositions_law.unweighted_rule_truth`).
UNWEIGHTED_RULE_TRUTH = 0.662385

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The nine calibration labels (plan section 5).
CALIBRATION_LABELS = law.LABELS
EFFICIENCY_BOUNDS = {label: LIMITS[label]["bound"] for label in CALIBRATION_LABELS}

#: The must-fail target controls (plan section 5): their own draws of the ``weighted`` setting,
#: each draw read twice. ``target_gap_control`` reads the unweighted :math:`\Psi_{d_0}(P_0) =
#: 0.577115`, 0.1130 below the target, 8.36 SEs at n = 2,000 with the weighted curve's SD
#: 0.604640: it fails if the weights are dropped. ``unweighted_rule_control`` reads
#: :math:`\Psi^w_{d_0}(P_w) = 0.662385`, 0.0277 below the target, 2.05 SEs: it fails if the
#: fit learns its rule without the weights. The design test computes each gap and each
#: control's power. Each passes when its exact 99% coverage interval lies wholly below the
#: calibration floor 0.92, declared before the pilot.
TARGET_FAMILY = "weighted_target_control"
TARGET_LABEL = "weighted"
TARGET_CONTROL = "target_gap_control"
RULE_CONTROL = "unweighted_rule_control"
TARGET_TRUTHS = {TARGET_CONTROL: "unweighted", RULE_CONTROL: "unweighted_rule"}

#: The in-sample lean (plan section 5, the T43 lean): a reporting reading, not a cell. The
#: ``in_sample`` calibration cell publishes its standardized bias, coverage and SE ratio against
#: :math:`\Psi_{d_n}`, the value of the one rule it learned, in ``rule_value_standardized_bias``
#: and its siblings (every cell publishes them); no verdict reads them.
LEAN_LABEL = "in_sample"

#: The construction control of each stratum label: the stratum curve without its
#: ``1 / P_{n,v}(W2 = s)`` normalisation, as ``learned-rule-strata`` declares it. Its SE is
#: about the stratum share (1/2) times the reported one, so it falls below the SE-ratio band by
#: construction: it shows that the instrument resolves a gross normalisation error.
UNNORMALISED = "unnormalised_curve_control"
STRATUM_LABELS = ("strata_s0", "strata_s1")


def calibration_cells() -> tuple[str, ...]:
    """Every ``interval_calibration`` cell, in label order."""
    cells: list[str] = []
    for label in CALIBRATION_LABELS:
        cells.extend(
            f"{label}__{kind}" for kind in (POSITIVE, "shrunken_se_control", "noise_control")
        )
        if label in STRATUM_LABELS:
            cells.append(f"{label}__{UNNORMALISED}")
    return tuple(cells)


PROPERTY_CELLS = {
    "interval_calibration": calibration_cells(),
    TARGET_FAMILY: tuple(f"{TARGET_LABEL}__{control}" for control in TARGET_TRUTHS),
}

#: The support rows the registering change adds (the plan's T55 rows), declared before any run
#: as the keyword arguments of :func:`cleverly._support_rows.point_row`: one P12 row per
#: composition, each with ``known_mechanism`` and the key dimension that names the composition
#: (fixed weighting, the ``strata`` or ``known_regimes`` qualifier, the in-sample splitting with
#: the learner families of condition C5a, the intermediate missingness), as the fits key them;
#: the design test requires every fitted key of every setting, both intermediate levels
#: included, to be covered by its row with equal options and qualifiers. A
#: row is added if, and only if, these verdicts pass as declared: the primary performance tests
#: of its setting's parameters, every ``interval_calibration`` verdict of its labels (positive
#: and both controls), and, for the weighted row, the ``weighted_target_control`` verdict.
#:
#: A red calibration row is a finite-sample red only when the 99% interval of its SE ratio
#: contains 1, its coverage is consistent with 0.95 and diagnosis finds no defect; it then goes
#: to ``reporting`` with an owner row, and its support row is not added. Any other red row is a
#: defect suspect. A control that passes is a design defect. No row is added for estimated
#: ``g``, which no cell measures.
SUPPORT_ROWS: dict[str, dict[str, Any]] = {
    "weighted": {
        "splitting": ("fold_evaluated",),
        "families": ("P12",),
        "weighting": "fixed",
        "qualifiers": ("known_mechanism",),
        "limits": (
            "It covers fixed observation weights with the weighted population's own treatment "
            "mechanism supplied as treatment_probabilities=; the cell measures the weight "
            "1 + A Y with g_w = g0 E[w | A, W] / E[w | W]. Supplying the design mechanism g0 "
            "beside weights that depend on the treatment or the outcome is the Fix-16 defect, "
            "which the key cannot see and no cell measures.",
        ),
    },
    "strata": {
        "splitting": ("fold_evaluated",),
        "families": ("P12",),
        "qualifiers": ("known_mechanism", "strata"),
    },
    "known_regimes": {
        "splitting": ("fold_evaluated",),
        "families": ("P12",),
        "qualifiers": ("known_mechanism", "known_regimes"),
    },
    "in_sample": {
        "splitting": ("in_sample",),
        "families": ("P12",),
        "qualifiers": ("known_mechanism",),
        "options": {
            "outcome_learner_family": "parametric",
            "treatment_learner_family": "known",
        },
    },
    "intermediate": {
        "splitting": ("fold_evaluated",),
        "families": ("P12",),
        "missing": ("intermediate",),
        "qualifiers": ("known_mechanism",),
    },
}

#: The truth-harness artifact: one row per primary replication and parameter.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Compositions of the value of the optimal rule",
    slug="learned-rule-optimal-compositions",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_optimal_compositions",
    document="docs/technical-reference/method-evidence/learned-rule-optimal-compositions.md",
    anchor="compositions-of-the-value-of-the-optimal-rule",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-optimal-compositions",
    reference=None,
    failure_bound={"cleverly-learned-rule-optimal-compositions": 0.01},
    modules=(
        "tests/studies/learned_rule_optimal_compositions.py",
        "tests/studies/learned_rule_optimal_compositions_properties.py",
        "tests/studies/_learned_rule_optimal_compositions_law.py",
        "tests/studies/_learned_rule_optimal_law.py",
        "tests/studies/_learned_rule_intermediate_law.py",
        "tests/studies/_learned_rule_in_sample_law.py",
        "tests/studies/_learned_rule_contrast_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/_fold_evaluated_strata_law.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/estimators/_strata.py",
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
    runner_module="tests.studies.learned_rule_optimal_compositions",
    properties_module="tests.studies.learned_rule_optimal_compositions_properties",
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
        "fit(..., treatment_probabilities=[1 - g(1 | W), g(1 | W)]) in every cell: RM30's "
        "logit g0(1 | W) = 0.3 W1 - 0.2 W2, except the weighted cell, which passes the "
        "weighted population's g_w(1 | W) = g0(1 | W)(1 + Qbar0(1, W)) / (1 + g0(1 | W) "
        "Qbar0(1, W))"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2), marked "
        "parametric in sample; on the intermediate law the same on (A, W1, W2, Z)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2), unused (known g)",
    "settings": {
        "weighted": "RM30 non_exceptional, weights='w' with w = 1 + A Y",
        "strata": "RM30 non_exceptional, strata=['W2']",
        "known_regimes": "RM30 non_exceptional, interventions=[Static(1), Static(0)]",
        "in_sample": "RM30 non_exceptional, cross_fit=False, cv_evaluation=False",
        "intermediate": (
            "the learned-rule-intermediate law, intermediate='Z', both levels; intermediate "
            "learner LogisticRegression(C=1e6, max_iter=1000) on (A, W1, W2)"
        ),
    },
    "truth": (
        "each label's parameter at its optimal rule (d0^w weighted, d0^z at a level, d0 "
        "otherwise) by scipy.integrate.quad over W1 for each W2, split at the rule's "
        "threshold; one number per label"
    ),
    "regret": (
        "per replication: refit each fold's outcome learner on the fit's training rows with the "
        "package's fit_on_rows, require the fit's rule on every validation row (rule L4), "
        "integrate each label's value at each fold rule by the trapezoid rule on 4001 points "
        "of W1 for each W2, and publish "
        "rule_value (the fold mean) and regret = truth - rule_value; descriptive"
    ),
    "rule_disagreement": "the record's share of rows on which the ten fold rules disagree",
    "initial_estimate": (
        "the fold average of the initial plug-ins Qbar_v(d_i, W_i) at the fit's own rule: "
        "weighted by the observation weights for the weighted value, within each W2 level for "
        "a stratum, and the rule's plug-in minus the regime's for a contrast"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed, with each parameter's fixed truth."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    truths = {name: law.truth(label) for label, name in law.PARAMETERS[scenario].items()}
    return law.draw(scenario, n, seed), truths


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, scenario, fold_seed)
    rows = []
    for label, name in law.PARAMETERS[scenario].items():
        estimate = fitted.estimate(label)
        std_error, low, high = reported_inference(estimate)
        truth = law.truth(label)
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
                "initial_estimate": law.untargeted_estimate(fitted, label),
                **law.harness_columns(fitted, label),
            }
        )
    return rows


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
    """The harness columns of every primary replication and parameter."""
    columns = ["scenario", "replicate", "estimand", "truth", *law.HARNESS_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "estimand", "replicate"], ignore_index=True
        )
    }
