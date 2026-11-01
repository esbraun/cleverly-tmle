r"""Registered evidence for the learned rule beside known regimes, and their contrasts (X11 (c)).

The subject is ``TMLE(learned_rule=LearnedRule(), interventions=[Static(1), Static(0)],
n_folds=10, targeting_scheme="pooled", g_bounds="auto")`` with RM30's interaction logistic
outcome learner and logistic treatment learner, and every other setting at the constructor's
default. The learned rule is column 0 of the fit's regime set and each known regime one more
column, on the same folds and the same fitted :math:`g_v`; the shipped regime fluctuation targets
every column. Each fit reports five parameters (``_learned_rule_contrast_law.TAGS``):

- the learned-rule value, whose target is the data-adaptive :math:`\tilde\psi_{0n}`;
- each known regime's value :math:`\Psi_a(P_0) = E_0\bar Q_0(a, W)`, a fixed target;
- each contrast :math:`\tilde\psi_{0n} - \Psi_a(P_0)`, with curve :math:`D_L - D_K` on the same
  rows and the shipped variance rule applied to it.

Both reports are read: the fold-evaluated one (``cv_evaluation=True``, the primary rows) and the
stacked one (``cv_evaluation=False``, X11 (e)) on the same draws and fold seeds. Two laws:

``non_exceptional``
    RM30's correctly specified law at n = 2,000. Every parameter is held to the framework's
    calibration bands (condition C4's first branch: both nuisances consistent).
``misspecified_limit``
    RM30's law whose blip the outcome model cannot represent, at n = 2,000, with the correct
    logistic treatment model fitted by maximum likelihood. The curves the fit reports omit the
    treatment model's projection term. Corollary 2 of van der Laan and Luedtke (2015) makes the
    variance from the curve conservative for an MLE of :math:`g` in a correct model, and the
    paper extends this to cross-validated TMLE by analogy only (``reviews/x11/X11-G-corollary3-
    reading.md``). On this law the omitted projection term is negligible (predicted SE ratio at
    most 1.0005, ``test_learned_rule_contrast_design``), so the cells measure coverage when the
    outcome model's treated arm is misspecified, not conservativeness. Its untreated arm is
    RM30's shared ``Qbar0(0, W)``, which the outcome model represents to within 0.005, so the
    ``known_0`` cells read a correctly specified arm. This law's cells claim the bias leg and
    coverage at or above the calibration floor, and no SE calibration.

The X11 wave-2 plan (``X11-wave2-plan.md`` rev 2.1 in the plans repository, section 4), its
review (finding 5), the implementation review ``reviews/x11/X11-G-impl-review.md`` (open
questions 1 to 6), the Corollary 3 reading and the study design record
``reviews/x11/X11-G-study-design.md`` declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_contrast_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

base = law.base

#: The primary replications: the fold-evaluated fit of ``non_exceptional``, all five
#: parameters. The primary gate reads the bias leg (the 99% interval of the mean error inside
#: 0.25 error SDs) of each. PRIMARY_REPLICATES is the smallest multiple of 400 of at least the
#: study's floor of 1,000 whose predictive pass probability of the five bias legs, on the
#: 4,000-draw pilot, is at least 0.95: the floor binds, since 800 already gives 0.999
#: (``test_learned_rule_contrast_design``).
PRIMARY_REPLICATES = 1_200
PRIMARY_N = 2_000
SEED = 20263091
RESAMPLING_SEED = 20263092

#: The primary law. ``misspecified_limit`` is read in the property phase only, because the
#: primary gate's SE sanity band is an SE gate, which Corollary 3's route does not license.
PRIMARY_LAW = "non_exceptional"
SCENARIOS = {PRIMARY_LAW: tuple(law.TAGS.values())}

#: The two reports, in the order every cell name reads them.
REPORTS = tuple(law.REPORTS)

#: The SD of ``sqrt(n)`` times each parameter's estimate at the limit rule of the declared
#: learner's population projection on ``non_exceptional``, by quadrature
#: (:func:`~tests.studies._learned_rule_contrast_law.efficiency_bounds`), for each calibration
#: label. ``value`` equals the RM30 bound 0.664438. They size the noise controls only, at one unit
#: each, and no verdict reads an efficiency ratio. ``test_learned_rule_contrast_design``
#: recomputes them.
TAG_BOUNDS = {
    "value": 0.664438,
    "known_1": 0.698330,
    "known_0": 0.688568,
    "contrast_1": 0.650857,
    "contrast_0": 0.708845,
}

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The calibration labels, one per report and parameter on ``non_exceptional``.
CALIBRATION_LABELS = tuple(
    f"{PRIMARY_LAW}_{report}_{tag}" for report in REPORTS for tag in law.TAGS
)
EFFICIENCY_BOUNDS = {
    f"{PRIMARY_LAW}_{report}_{tag}": TAG_BOUNDS[tag] for report in REPORTS for tag in law.TAGS
}

#: The ``misspecified_limit`` family: each report's five parameters, gated on the bias leg and
#: on the coverage interval's lower bound at or above the calibration floor 0.92, with no SE
#: ratio gate (Corollary 3's route, the module docstring). Each positive's control is the same
#: estimate with its SE shrunk by :data:`COVERAGE_SHRINK`, whose coverage interval must fall
#: wholly below 0.92.
COVERAGE_FAMILY = "conservative_coverage"
COVERAGE_LAW = "misspecified_limit"
COVERAGE_LABELS = tuple(f"{COVERAGE_LAW}_{report}_{tag}" for report in REPORTS for tag in law.TAGS)
COVERAGE_CONTROL = "shrunken_coverage_control"

#: The shrink of the coverage family's control, declared before the pilot was read. The claim
#: admits any over-coverage, so the control must fail at every SE ratio the claim admits: at a
#: true ratio of 1.30 the shrunken interval covers about 0.87 (``2 Phi(1.96 * 0.6 * 1.3) - 1``),
#: below the floor by more than the 99% half-width at the declared count.
COVERAGE_SHRINK = 0.60

#: The covariance control of each contrast: the same estimate with ``|se_L - se_K|``, the SE the
#: contrast would have if the learned value and the known value had correlation one. It must
#: under-cover: its coverage interval must lie wholly below :data:`CORRELATION_ONE_CEILING`.
#: Every contrast of both laws and both reports carries one.
COVARIANCE_FAMILY = "contrast_covariance"
CORRELATION_ONE = "correlation_one_control"

#: The must-fail threshold of the correlation-one control, recorded after the pilot and before
#: any declared draw (impl review open question 4). A must-fail control passes when its coverage
#: interval lies below the ceiling, so a lower ceiling is harder to pass. The pre-pilot rule
#: would be the claim's own floor 0.92, the rule of the coverage family's control; 0.80 lies
#: below it, so choosing it after the pilot cannot have made a verdict easier. The value is the
#: framework's must-fail convention (``FEW_CLUSTER_IID_CONTROL_CEILING`` and
#: ``CLUSTER_ROBUST_CONTROL_SE_CEILING``). The pilot read 0.052 to 0.086 on the eight cells,
#: every 99% upper bound below 0.10, which checks that the control has power at either ceiling.
CORRELATION_ONE_CEILING = 0.80

#: The diagnostic of each contrast: the same estimate with ``sqrt(se_L^2 + se_K^2)``, the SE if
#: the two estimates were independent. Role ``diagnostic``, no verdict: it is expected to
#: over-cover, and the summary publishes its SE ratio and coverage.
INDEPENDENT_FAMILY = "independent_se_diagnostic"
INDEPENDENT = "independent_se"

#: The bias control (plan section 4): the fold-evaluated fit with RM30's ``drop_w1_learner``,
#: read on the contrast against ``Static(0)``. The positive arm is the targeted contrast; the
#: control is the same fit's untargeted contrast, the difference of the initial plug-ins. The
#: contrast against ``Static(1)`` is not read: the learner's limit rule treats every row, which
#: is ``Static(1)``, so that contrast's target and curve are zero on every fold whose rule settled
#: (the design record).
NECESSITY_FAMILY = "targeting_necessity"
NECESSITY_LABEL = "drop_w1_contrast_0"
NECESSITY_TAG = "contrast_0"
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

#: The support rows the registering change adds, declared before any run. Each is a P11 row
#: with the qualifier ``known_regimes`` and the report's splitting, and covers every parameter a
#: contrast fit reports (``ey_learned_rule``, ``ey_known_regime``, ``ate_learned_rule``), because
#: a row labels every estimate of a fit (impl review open question 2).
#:
#: The fold-evaluated row is added if, and only if, these verdicts pass as declared: the primary
#: performance tests of all five parameters (bias leg, coverage floor, SE sanity); every
#: ``interval_calibration`` verdict of the ``fold_evaluated`` labels, positives and controls; every
#: ``conservative_coverage`` verdict of the ``fold_evaluated`` labels; every ``contrast_covariance``
#: verdict of the ``fold_evaluated`` cells; and the ``targeting_necessity`` verdicts.
#:
#: The stacked row is added if, and only if, the same verdicts pass with the ``stacked`` labels in
#: place of the ``fold_evaluated`` ones, and the primary and necessity verdicts pass too. At ten
#: equal folds without weights the stacked point is the fold-evaluated point (pooled targeting,
#: ``W_v/W = 1/V``), so the fold-evaluated primary bias legs and the necessity pair read the
#: stacked point as well; only the variance rules differ (the design record).
#:
#: A red primary bias leg, a red positive calibration or coverage cell, or a control that does
#: not fail is diagnosed before any route is chosen. With no defect found, the cell is published
#: red under ``reporting`` with an owner row, and the row of that report is not added. Contrast
#: fits outside the rows stay experimental.
SUPPORT_ROWS = {
    "fold_evaluated": {"splitting": ("fold_evaluated",), "qualifiers": ("known_regimes",)},
    "stacked": {"splitting": ("stacked",), "qualifiers": ("known_regimes",)},
}

#: The truth-harness artifact: one row per primary replication and parameter.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned-rule contrasts with known regimes",
    slug="learned-rule-contrast",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_contrast",
    document="docs/technical-reference/method-evidence/learned-rule-contrast.md",
    anchor="learned-rule-contrasts-with-known-regimes",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-contrast",
    reference=None,
    failure_bound={"cleverly-learned-rule-contrast": 0.01},
    modules=(
        "tests/studies/learned_rule_contrast.py",
        "tests/studies/learned_rule_contrast_properties.py",
        "tests/studies/_learned_rule_contrast_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/interventions/base.py",
        "src/cleverly/targets/builtin.py",
        "src/cleverly/learners/crossfit.py",
        "src/cleverly/_support.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_contrast",
    properties_module="tests.studies.learned_rule_contrast_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(), interventions=[Static(1), Static(0)], n_folds=10, "
        "cv_evaluation=True, targeting_scheme='pooled', g_bounds='auto', "
        "random_state=<fold seed>) at n = 2000; every other setting the constructor's default "
        "(simultaneous=True, whose band the fit skips); repeats=1"
    ),
    "stacked_fit": "the same with cv_evaluation=False, on the same draw and fold seed",
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "parameters": dict(law.TAGS),
    "laws": {
        "non_exceptional": "RM30 law with blip 0.1 + W1; calibration bands",
        "misspecified_limit": (
            "RM30 law with blip 0.8 W1 + W1^2 - 0.3; bias leg and coverage floor 0.92 only "
            "(Corollary 3's route, by the paper's analogy); the omitted projection term is "
            "negligible here (predicted SE ratio at most 1.0005), so the cells measure coverage "
            "under a misspecified treated arm, not conservativeness"
        ),
    },
    "variance": (
        "fold-evaluated: V^-2 sum_v n_v^-2 sum_{i in v} D_i^2 of each curve, D_L - D_K for a "
        "contrast; stacked: the centred rule {n (n - 1)}^-1 sum_i (phi_i - phibar)^2 of the "
        "stored curve, the difference of the two curves for a contrast"
    ),
    "truth": (
        "per replication: refit each fold's outcome learner on the fit's training rows with "
        "the package's fit_on_rows, require the fit's rule on every validation row (rule L4), "
        "integrate each fold rule's value by the trapezoid rule on 4001 points of W1 for each "
        "W2 with the known Qbar0, and weight fold v by the report's fold weight (1/V, or n_v/n "
        "stacked), checked against the record's fold_weights; a known regime's truth is "
        "E Qbar0(a, W) by the same rule; a contrast's is the difference"
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
        "RM30's drop_w1_learner, fold-evaluated, contrast against Static(0): targeted against "
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
    result = fitted.results["fold_evaluated"]
    truths = law.truths(result, fitted.refit, scenario)
    rows = []
    for tag, name in law.TAGS.items():
        estimate = result.estimates[name]
        std_error, low, high = reported_inference(estimate)
        truth, oracle = truths[tag]
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
