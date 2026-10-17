"""Registered evidence for the per-arm outcome-adaptive C-TMLE with strata and missing data.

The subject is ``CTMLE(strategy="oat")`` at its default ``oat_design="per_arm"``, fitted with
baseline strata (``strata=``), a missing outcome (``delta=``) or a missing treatment
(``treatment_delta=``).  For arm ``a`` and ``q_a = Qbar_n(a, W)`` the design is the product of
the treatment factor ``P(A = a | Delta_A = 1, q_a)``, the outcome-observation factor
``P(Delta = 1 | A = a, Delta_A = 1, q_a)`` and the treatment-observation factor
``P(Delta_A = 1 | q_a)``, each fitted inside each stratum.  By the chain rule the product is
``P(C_a = 1 | q_a)`` for ``C_a = Delta_A Delta 1{A = a}``.  The construction is Benkeser, Cai
and van der Laan (2020), Section 3, applied to ``C_a`` (``X29-plan.md``, section 1.3).

**Points only.**  Every ``oat`` fit has the status ``generated_design_plugin``: it withholds
its interval, and the study reads ``plugin_std_error`` and ``plugin_interval`` through
:func:`~tests.studies.evidence.schema.reported_inference`.  The study claims consistency of the
point estimate.  It claims no interval.

**The laws** are those of :mod:`tests.studies.oat_per_arm_extension_laws`.

**The comparator** is R ``drtmle`` 1.1.2 at ``538a3a2`` by the X17 binary recode per arm, with
``DeltaY``, ``DeltaA`` and ``stratify = TRUE``, through
``tests/canonical/drtmle_oat_per_arm_strata_missing/run_drtmle_oat_per_arm_strata_missing.R``.
For the ``strata`` primary the same call runs on each stratum's rows, and the runner forms the
``P_n(S = s)`` mixture of the stratum points and its curve in R.  The runner rebuilds each TMLE
curve on its submodel and refuses a fit whose curve does not reproduce ``tmle$cov[1, 1]`` to
``1e-8``, as the X17 runner does.  Every primary estimand is paired.  Natural-course
parameters have no ``drtmle`` counterpart under ``adapt_g``, so they appear in property cells
only.

**What the Python rows also carry.**  ``shipped_estimate`` is the shipped construction on the
same fit: the per-arm treatment factor fitted on every row, without strata, for the
``strata`` primary; and the treatment factor times the full-covariate ``pi(a, W)`` of
``nuisance.tilt_missingness`` for the ``missing_outcome`` primary.  :func:`longhand` builds it
from the fit's own nuisances; it is not a code path of the package.  ``longhand_gap`` is the
distance between :func:`longhand` with the fit's own design and the fit's point, a check of the
longhand.  ``rows_at_bound`` counts the rows at which any factor of the fit sits at its bound.

**The declared reading rule** is :data:`READING_RULE`, and :func:`defect_suspects` applies it.
The post-run gate of
``tests/unit/test_ctmle_oat_per_arm_strata_missing_study_design.py`` reads the committed
artefacts through it, and it skips until they exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from cleverly.estimators import CTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import oat_per_arm_extension_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import primary_rows

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
#: The primary size and budget.  Condition (g) sets the size: the smallest stratum of the
#: ``strata`` primary holds at least 400 rows with probability 0.9965 at ``n = 2,250``.
#: Condition (r) sets the budget: a cell whose true standardized bias is 0.10 passes the
#: bias rule with probability 0.950 at ``R = 800``.
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_250
SEED = 20261029
RESAMPLING_SEED = 2026102901
IMPLEMENTATION = "cleverly-ctmle-oat-per-arm"
REFERENCE = "drtmle-r-oat-per-arm-strata-missing"
STRATIFY_FOLDS = "none"
NUISANCE_BOUND = laws.NUISANCE_BOUND

#: The stratum labels the ``strata`` primary reports, as the fit names them.
STRATUM_LABELS = laws.STRATA_COMPLETE.stratum_labels
STRATA_ESTIMANDS = tuple(
    f"{name}{suffix}"
    for name in ("ey0", "ey1", "ate")
    for suffix in ("", *(f"[{label}]" for label in STRATUM_LABELS))
)
BINARY_ESTIMANDS = ("ey0", "ey1", "ate", "rr", "or")
THREE_ARM_ESTIMANDS = (
    "ey[0]",
    "ey[1]",
    "ey[2]",
    "ate[1 vs 0]",
    "ate[2 vs 0]",
    "rr[1 vs 0]",
    "rr[2 vs 0]",
    "or[1 vs 0]",
    "or[2 vs 0]",
)
#: Scenario -> the law it samples.
SCENARIO_LAWS: Mapping[str, laws.ExtensionLaw] = {
    "strata": laws.STRATA_COMPLETE,
    "missing_outcome": laws.MISSING_OUTCOME,
    "three_arm_missing": laws.THREE_ARM_MISSING,
    "missing_treatment": laws.MISSING_TREATMENT,
}
SCENARIO_ESTIMANDS: Mapping[str, tuple[str, ...]] = {
    "strata": STRATA_ESTIMANDS,
    "missing_outcome": BINARY_ESTIMANDS,
    "three_arm_missing": THREE_ARM_ESTIMANDS,
    "missing_treatment": BINARY_ESTIMANDS,
}
#: The scenarios whose rows carry the shipped construction, for the reported design gaps.
SHIPPED_SCENARIOS = ("strata", "missing_outcome")


#: The labels of the multi-estimand property cells, ``<estimand slug>__<configuration>``.
def _slug(name: str) -> str:
    base, _, rest = name.partition("[")
    if not rest:
        return base
    return f"{base}_s{rest.rstrip(']').split('=')[1]}"


def multi_cells(configuration: str, estimands: Sequence[str]) -> tuple[str, ...]:
    """The published cell names of one multi-estimand configuration, in estimand order."""
    return tuple(f"{_slug(name)}__{configuration}" for name in estimands)


#: The estimands each multi-estimand property configuration reads, as the fit names them.
#: :mod:`tests.studies.ctmle_oat_per_arm_strata_missing_properties` fits each once per
#: replication and writes one row per estimand.
STRATUM_SUFFIXES = ("", *(f"[{label}]" for label in STRATUM_LABELS))
PROPERTY_ESTIMANDS: Mapping[str, tuple[str, ...]] = {
    "natural_course_insample": ("ey_obs", "par", "paf"),
    "missing_outcome_crossfit": ("ey0", "ey1", "ate", "rr", "or", "ey_obs", "par", "paf"),
    "strata_missing": tuple(
        f"{name}{suffix}" for name in ("ate", "ey_obs", "par", "paf") for suffix in STRATUM_SUFFIXES
    ),
    "weighted_strata_missing": tuple(f"ate{suffix}" for suffix in STRATUM_SUFFIXES),
    "strata_missing_outcome_wrong": ("ate",),
}
#: The ``root_n_and_efficiency`` rungs.  Re-declared after the first run, which had
#: ``(500, 2_000, 8_000)``: :data:`LADDER_DECLARATION` records why.
LADDER = (2_000, 4_000, 8_000)

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "robustness_contract": tuple(
        cell
        for configuration, estimands in PROPERTY_ESTIMANDS.items()
        for cell in multi_cells(configuration, estimands)
    ),
    "root_n_and_efficiency": tuple(f"n_{n}" for n in LADDER),
    "root_n_rate": ("empirical_sd", "reported_se"),
}

#: The owner of every red row the reading rule routes, by its ``id`` in the roadmap: F19, the
#: generated-design term of the per-arm design.
RED_CELL_OWNER = "F19"
#: The upper end of the 99% SE-ratio interval a routed row must stay below.  The X17 term makes
#: the plug-in standard error short, not long, so a long one is not routed.
ROUTED_SE_RATIO_CEILING = 1.07
#: The lowest point SE ratio a routed row may have.  Added after the run, as a strengthening:
#: a standard error far shorter than the X17 term makes it is a defect suspect, not F19's.  The
#: lowest routed row, ``three_arm_missing`` ``ey[1]`` of both implementations, reads 0.736, so
#: the floor changes no routing of the committed run.
ROUTED_SE_RATIO_FLOOR = 0.70

#: The reading rule, declared before the run, with the floor
#: :data:`ROUTED_SE_RATIO_FLOOR` added after it.  :func:`defect_suspects` applies it to the
#: committed verdict tables.
READING_RULE = (
    "Defect suspects (diagnose, fix, re-run under a fresh declaration): a truth row whose bias "
    "component fails; any red paired row; any red property or control row other than a "
    "routed ladder rung; a failed replicate; and a truth row red on coverage or the SE ratio "
    "whose SE ratio is not short. Routed to F19 in red_cells.py OWNERS, as X17's rows are: a "
    f"truth row of {IMPLEMENTATION} or of {REFERENCE} that is red only on its coverage or "
    "SE-ratio reading, with bias_equivalent True, a point SE ratio below 1, and the upper end "
    f"of its 99% SE-ratio interval below {ROUTED_SE_RATIO_CEILING}. A root_n_and_efficiency "
    "rung reads the same way: red only on its coverage or SE-ratio reading, with "
    "bias_equivalent True, a point SE ratio below 1 and the upper end of its 99% bootstrap "
    f"SE-ratio interval below {ROUTED_SE_RATIO_CEILING}, it is routed to F19; red on its bias, "
    "it is a defect suspect. Added after the run, as a strengthening that changes no routing "
    "of the committed run: a routed truth row or rung also needs a point SE ratio of at least "
    f"{ROUTED_SE_RATIO_FLOOR:.2f}, and a shorter one is a defect suspect (the lowest routed "
    "row, three_arm_missing ey[1], reads 0.736)"
)

STUDY = StudyRecord(
    name="outcome-adaptive per-arm C-TMLE with strata and missing data (point estimates)",
    slug="ctmle-oat-per-arm-strata-missing",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_oat_per_arm_strata_missing",
    document=(
        "docs/technical-reference/method-evidence/outcome-adaptive-per-arm-c-tmle-strata-missing.md"
    ),
    anchor="outcome-adaptive-per-arm-c-tmle-with-strata-and-missing-data",
    scenarios=dict(SCENARIO_ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    failure_bound={IMPLEMENTATION: 0.01, REFERENCE: 0.01},
    comparison="matching implementation",
    comparison_overrides={
        "strata:*": "related target",
    },
    extra_artifacts=("design-gap.csv", "fit-diagnostics.csv"),
    modules=(
        "tests/studies/canonical_ctmle_oat_per_arm_strata_missing.py",
        "tests/studies/ctmle_oat_per_arm_strata_missing_properties.py",
        "tests/studies/oat_per_arm_extension_laws.py",
        "tests/studies/oat_per_arm_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/drtmle/Dockerfile",
        "tests/canonical/drtmle_oat_per_arm_strata_missing/run_drtmle_oat_per_arm_strata_missing.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_ctmle_oat_per_arm_strata_missing",
    properties_module="tests.studies.ctmle_oat_per_arm_strata_missing_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
    calibration_efficiency_ratio=False,
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "per arm a: drtmle(W, A = as.numeric(!is.na(A) & DeltaA == 1 & A == a), "
        "Y = ifelse(DeltaY == 1, Y, NA), DeltaY, DeltaA, a_0 = c(1, 0), adapt_g = TRUE, "
        "stratify = TRUE, Qn = list(qa, qa), glm_g = list(A = B, DeltaY = B, DeltaA = B) with B "
        "the cubic in qlogis(pmin(pmax(Q1W, 1e-6), 1 - 1e-6)), cvFolds = 1, guard = NULL, "
        "targeted_se = TRUE, tolg = 1e-4, maxIter = 100, tolIC = 1e-10); qa is this package's "
        "initial outcome regression; on the strata scenario the same call on each stratum's "
        "rows; tmle$est[1] and tmle$cov[1, 1] give the arm mean and its variance; each TMLE "
        "curve is rebuilt on the logistic submodel at the reported estimate and checked "
        "against tmle$cov[1, 1] to 1e-8; contrasts by the delta method on the stacked curves; "
        "the stratum curves embedded as 1{S = s} / P_n(S = s) times the subset curve, and the "
        "marginal as the P_n(S = s) mixture with its mass term"
    ),
    "red_cell_owner": RED_CELL_OWNER,
    "reading_rule": READING_RULE,
}

#: The budget measured before the declaration, in single-process CPU seconds per fit on
#: replications 0 to 1,999 of each cell's declared stream.  :data:`CONFIGURATION` quotes it.
#: The ``n_4000`` rung was measured with the re-declared ladder on a second machine, which
#: also re-measured ``n_2000`` and ``n_8000``; the ratio of the two machines scales it.
MEASURED_BUDGET = (
    "Python, CPU seconds per fit: primaries (fit, longhand and shipped refit) strata 0.099, "
    "missing_outcome 0.057, three_arm_missing 0.074, missing_treatment 0.069; properties "
    "natural_course_insample 0.049, missing_outcome_crossfit 0.177, strata_missing 0.368, "
    "weighted_strata_missing 0.579, strata_missing_outcome_wrong 0.353, n_2000 0.357, "
    "n_8000 0.470, and the first run's dropped rung n_500 0.338. The re-declared rung "
    "n_4000 was measured on a second machine, on replications 0 to 1,999: 0.293, against "
    "0.259 for n_2000 and 0.359 for n_8000 there, so about 0.39 on the first machine. A "
    "20-replicate smoke of n_4000 there took 6.8 CPU seconds (0.342 per fit). The declared "
    "run is 3,200 primary and 6,400 property fits, about 2,400 CPU seconds (0.7 CPU-hours) "
    "before the shared summaries. R: 13 drtmle calls per primary replication, 10,400 calls, "
    "not measured here (plan estimate 0.6 to 1.5 CPU-hours)"
)
FAILURE_PROBE = (
    "replications 0 to 1,999 of every cell and every primary scenario (24,000 fits, 5,982 "
    "CPU seconds on three processes) raised nothing. Targeting ended above tol = 1e-10 "
    "with a ConvergenceWarning in n_500 400, n_2000 55, n_8000 7, strata_missing 31, "
    "weighted_strata_missing 45 and strata_missing_outcome_wrong 25 of 2,000 fits, and in "
    "no primary fit; such a fit returns its estimate and counts as a replication. The "
    "re-declared rung n_4000 was probed the same way on replications 0 to 1,999: no fit "
    "raised, and 19 of 2,000 ended with a ConvergenceWarning; the same probe re-read n_2000 "
    "55 and n_8000 7. The R reference was not probed"
)
#: The ladder's re-declaration, quoted by :data:`CONFIGURATION`.
#: :mod:`tests.studies.ctmle_oat_per_arm_strata_missing_properties` gives the reason.
LADDER_DECLARATION = (
    "root_n_and_efficiency at n = 2,000, 4,000 and 8,000, R = 800 each (condition r: pass "
    "probability 0.950 at d = 0.10). Re-declared after the first run (commit ae53bbb), whose "
    "ladder was n = 500, 2,000, 8,000. Its n_500 rung read a mean SE ratio of 1.575 (mean "
    "plug-in SE 0.0884, empirical SD 0.0561), a median SE of 1.00 empirical SDs, coverage "
    "0.954 and an equivalent bias; 40 of 800 fits (5%) had an SE above twice the empirical "
    "SD, and without them the mean ratio was 1.035. The reported_se slope read -0.71. The "
    "diagnosis found no code defect: each factor was fitted on about 20 to 35 rows per "
    "stratum, arm and fold, and the unpenalized cubic factor learner extrapolated the design "
    "toward zero at one or two rows, whose term dominated the plug-in variance. The rung "
    "measured learner instability in small cells, not the rate, and was dropped. Condition "
    "t, added with the re-declaration: the smallest factor training cell (stratum x arm x "
    "fold training complement, recorded rows) holds at least 100 rows in expectation at the "
    "smallest rung; 117.1 at n = 2,000 (S = 2, arm 0, share 0.0732, four fifths in training), "
    "29.3 at n = 500. The rerun of the property phase supersedes the first run's property "
    "artifacts; the primaries, the pairing, the robustness cells, the laws, the seeds and the "
    "reading rule are unchanged. Condition t restated after the rerun, without a new floor: a "
    "factor fit is bound by the smaller class of its binary target, and on the strata law that "
    "is the nonrespondents of arm 0 in S = 2. The expected count of that class per stratum, "
    "arm, fold and factor (oat_per_arm_extension_laws.smallest_minority_class) is 9.9 at "
    "n = 500, 39.5 at n = 2,000, 59.2 in the strata_missing cells at n = 3,000, 79.0 at "
    "n = 4,000 and 157.9 at n = 8,000. The declared ladder does not meet the 100-row floor on "
    "that count, and the measured consequence is a tail of 3 in 800 fits at n = 2,000 and of 2 "
    "in 800 in the strata_missing cells, with a standard error above twice the empirical SD"
)

#: The revision of ``properties.csv`` after the rerun, quoted by :data:`CONFIGURATION`.
#: :mod:`tests.studies.ctmle_oat_per_arm_strata_missing_properties` gives the reason.
PROPERTY_SUMMARY_REVISION = (
    "properties.csv was summarized again from the committed property-replicates.csv.gz, "
    "with no new fit, after the rerun: the summary now multiplies the log-scale "
    "plugin_std_error of each rr and or row by its estimate (the delta method), so each SE "
    "ratio compares a standard error and an empirical SD of psi. The rerun's first summary "
    "compared the log-scale standard error with the SD of psi (missing_outcome_crossfit SE "
    "ratio 0.559 for rr and 0.322 for or). No verdict changed. study_module_sha256 records the "
    "modules of the run that fitted the replicates"
)


CONFIGURATION = {
    "strategy": "oat",
    "oat_design": "per_arm",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "inference": (
        "points only: every oat fit is generated_design_plugin, and each row reads "
        "plugin_std_error and plugin_interval"
    ),
    "g_bounds": list(laws.G_BOUNDS),
    "nuisance_bound": NUISANCE_BOUND,
    "q_bounds": (
        "the default, None: the plan named (0, 1), and the estimator refuses q_bounds on a "
        "binary outcome, whose bounds are (0, 1) already"
    ),
    "stratify_folds": STRATIFY_FOLDS,
    "outcome_learner": (
        "unpenalized logistic regression on [A indicators, W, S indicators, A x W, A x S]"
    ),
    "treatment_learner": (
        "per arm, factor and stratum, unpenalized logistic regression of the factor's target "
        "on (l, l^2, l^3), l = logit(clip(Qbar(a, W), 1e-6, 1 - 1e-6))"
    ),
    "missingness_learner": (
        "unpenalized logistic regression on [A, W] (and S): the full-covariate model "
        "tilt_missingness holds, which only the reported shipped construction reads"
    ),
    "laws": {name: law.name for name, law in SCENARIO_LAWS.items()},
    "reading_rule": READING_RULE,
    "failure_probe": FAILURE_PROBE,
    "measured_budget": MEASURED_BUDGET,
    "ladder": LADDER_DECLARATION,
    "property_summary_revision": PROPERTY_SUMMARY_REVISION,
}


def roles(law: laws.ExtensionLaw) -> dict[str, Any]:
    """The fit's role arguments for a sample of ``law``."""
    out: dict[str, Any] = {"outcome": "Y", "treatment": "A"}
    if law.missing_outcome:
        out["delta"] = "Delta"
    if law.missing_treatment:
        out["treatment_delta"] = "DA"
    if law.stratified:
        out["strata"] = ["S"]
    if law.weighted:
        out["weights"] = "wt"
    return out


def estimator(
    law: laws.ExtensionLaw,
    estimands: Sequence[str],
    *,
    cross_fit: bool = False,
    outcome: str = "correct",
) -> CTMLE:
    """The study's subject for a sample of ``law``.

    Parameters
    ----------
    law : ExtensionLaw
        The law the sample comes from.
    estimands : Sequence of str
        The estimand requests.
    cross_fit : bool, optional
        Whether to cross-fit with five unstratified folds.
    outcome : str, optional
        ``"correct"`` for the correctly specified outcome model, ``"wrong"`` for X17's model
        without ``W1``.

    Returns
    -------
    CTMLE
        The unfitted estimator.
    """
    strata = len(law.strata)
    outcome_learner = (
        laws.confounder_omitted_outcome(law.k, strata)
        if outcome == "wrong"
        else laws.extension_outcome(law.k, strata)
    )
    return CTMLE(
        strategy="oat",
        oat_design="per_arm",
        outcome_learner=outcome_learner,
        treatment_learner=laws.cubic_logit_learner(),
        missingness_learner=laws.missingness_glm(),
        cross_fit=cross_fit,
        n_folds=5,
        estimands=tuple(estimands),
        reference="0" if law.k > 2 else None,
        simultaneous=False,
        g_bounds=laws.G_BOUNDS,
        nuisance_bound=NUISANCE_BOUND,
        stratify_folds=STRATIFY_FOLDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    """Fit this study's subject, in sample, on one sample of ``scenario``'s law."""
    law = SCENARIO_LAWS[scenario]
    requests = ("ey", "ate", "rr", "or") if law.k > 2 else ("ey0", "ey1", "ate", "rr", "or")
    if scenario == "strata":
        requests = ("ey0", "ey1", "ate")
    return estimator(law, requests).fit(frame, **roles(law)).single()


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIO_LAWS:
        raise KeyError(scenario)
    frame, truth = SCENARIO_LAWS[scenario].sample(n, seed)
    return frame, {name: truth[name] for name in STUDY.scenarios[scenario]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ----------------------------------------------------------------------------- the design


def design(result: Any) -> np.ndarray:
    """The fit's realised design ``G``, ``(n, K)``: the bounded factor product of each arm.

    The treatment factor is clipped to ``g_bounds`` and each observation factor to
    ``[nuisance_bound, 1]``, as the fit bounds them.
    """
    nuisance = result.nuisance
    product = np.clip(np.asarray(nuisance.propensity.values, dtype=float), *laws.G_BOUNDS)
    for observation in (nuisance.missingness, nuisance.treatment_observation):
        if observation is not None:
            product = product * np.clip(np.asarray(observation, dtype=float), NUISANCE_BOUND, 1.0)
    return product


def rows_at_bound(result: Any) -> int:
    """The rows at which any factor of any arm sits at its bound."""
    nuisance = result.nuisance
    lower, upper = laws.G_BOUNDS
    treatment = np.asarray(nuisance.propensity.values, dtype=float)
    hit = np.any((treatment <= lower) | (treatment >= upper), axis=1)
    for observation in (nuisance.missingness, nuisance.treatment_observation):
        if observation is not None:
            hit |= np.any(np.asarray(observation, dtype=float) <= NUISANCE_BOUND, axis=1)
    return int(np.sum(hit))


def _fluctuate(
    offset: np.ndarray, covariate: np.ndarray, outcome: np.ndarray, rows: np.ndarray
) -> float:
    """The logistic fluctuation coefficient that solves the score on ``rows``, by Newton."""
    epsilon = 0.0
    for _ in range(200):
        fitted = expit(offset[rows] + epsilon * covariate[rows])
        score = float(np.sum(covariate[rows] * (outcome[rows] - fitted)))
        information = float(np.sum(covariate[rows] ** 2 * fitted * (1.0 - fitted)))
        step = score / information
        epsilon += step
        if abs(step) < 1e-14:
            return epsilon
    raise RuntimeError("the longhand fluctuation did not converge")


def longhand(result: Any, design_matrix: np.ndarray, *, stratified: bool) -> dict[str, float]:
    """The per-arm TMLE of every arm mean, from the fit's ``Qbar`` and a given design.

    One logistic fluctuation per arm, and per stratum when ``stratified``, with the
    covariate ``1 / G`` on the rows with ``C_a = 1``.  The arm mean is the mean of the
    targeted regression over every row; a stratum mean is its mean over the stratum's rows.

    Parameters
    ----------
    result : Any
        The fit, whose initial outcome regression and data the longhand reads.
    design_matrix : ndarray
        ``(n, K)``, the design of each arm.
    stratified : bool
        Whether to fluctuate inside each stratum.

    Returns
    -------
    dict
        Each arm mean, and with strata each stratum arm mean, by its arm position: ``"0"``,
        ``"0|S=1"`` and so on.
    """
    data = result.data
    treatment = np.asarray(data.treatment, dtype=float)
    recorded = np.asarray(data.treatment_recorded, dtype=bool)
    observed = np.asarray(data.observed, dtype=bool)
    outcome = np.nan_to_num(np.asarray(data.outcome, dtype=float))
    strata = np.asarray(data.strata) if stratified else np.zeros(data.n, dtype=int)
    labels = {code: data.stratum_label(code) for code in range(data.n_strata)} if stratified else {}
    out: dict[str, float] = {}
    for j, code in enumerate(data.arm_codes):
        q = np.asarray(result.nuisance.outcome.arms[code], dtype=float)
        hit = recorded & observed & (treatment == code)
        covariate = 1.0 / design_matrix[:, j]
        offset = logit(q)
        targeted = np.empty(data.n)
        for stratum in np.unique(strata):
            region = strata == stratum
            epsilon = _fluctuate(offset, covariate, outcome, hit & region)
            targeted[region] = expit(offset[region] + epsilon * covariate[region])
            if stratified:
                out[f"{j}|{labels[int(stratum)]}"] = float(np.mean(targeted[region]))
        out[str(j)] = float(np.mean(targeted))
    return out


def named(arms: Mapping[str, float], scenario: str) -> dict[str, float]:
    """The study's estimand names from :func:`longhand`'s arm means."""

    def odds(p: float) -> float:
        return p / (1.0 - p)

    out: dict[str, float] = {}
    suffixes = [""]
    if scenario == "strata":
        suffixes += [f"|{label}" for label in STRATUM_LABELS]
    for suffix in suffixes:
        tail = f"[{suffix[1:]}]" if suffix else ""
        k = SCENARIO_LAWS[scenario].k
        means = [arms[f"{j}{suffix}"] for j in range(k)]
        if k == 2:
            out[f"ey0{tail}"] = means[0]
            out[f"ey1{tail}"] = means[1]
            out[f"ate{tail}"] = means[1] - means[0]
            out[f"rr{tail}"] = means[1] / means[0]
            out[f"or{tail}"] = odds(means[1]) / odds(means[0])
            continue
        for j in range(k):
            out[f"ey[{j}]"] = means[j]
        for j in range(1, k):
            pair = f"{j} vs 0"
            out[f"ate[{pair}]"] = means[j] - means[0]
            out[f"rr[{pair}]"] = means[j] / means[0]
            out[f"or[{pair}]"] = odds(means[j]) / odds(means[0])
    return out


def shipped_design(result: Any, scenario: str) -> np.ndarray:
    """The shipped construction's design, built locally from the fit's own nuisances.

    ``strata``: the per-arm treatment factor fitted on every row, without strata, by the
    study's treatment learner on ``q_a``.  ``missing_outcome``: the fit's treatment factor
    times the full-covariate ``pi(a, W)`` of ``nuisance.tilt_missingness``, the product the
    package shipped before X29.  Neither is a code path of the package.
    """
    data = result.data
    if scenario == "strata":
        treatment = np.asarray(data.treatment, dtype=float)
        columns = []
        for code in data.arm_codes:
            q = np.asarray(result.nuisance.outcome.arms[code], dtype=float)
            model = laws.cubic_logit_learner().fit(
                q.reshape(-1, 1), (treatment == code).astype(float)
            )
            columns.append(model.predict_proba(q.reshape(-1, 1))[:, 1])
        return np.clip(np.column_stack(columns), *laws.G_BOUNDS)
    if scenario == "missing_outcome":
        treatment_factor = np.clip(
            np.asarray(result.nuisance.propensity.values, dtype=float), *laws.G_BOUNDS
        )
        tilt = np.clip(
            np.asarray(result.nuisance.tilt_missingness, dtype=float), NUISANCE_BOUND, 1.0
        )
        return treatment_factor * tilt
    raise ValueError(f"no shipped construction is reported for {scenario!r}")


def initials(result: Any, scenario: str) -> dict[str, float]:
    """The untargeted plug-in of each estimand, as the R runner forms it from ``Qn``."""
    data = result.data
    q = np.column_stack(
        [np.asarray(result.nuisance.outcome.arms[code], dtype=float) for code in data.arm_codes]
    )
    arms = {str(j): float(np.mean(q[:, j])) for j in range(q.shape[1])}
    if scenario == "strata":
        strata = np.asarray(data.strata)
        for code in range(data.n_strata):
            label = data.stratum_label(code)
            region = strata == code
            for j in range(q.shape[1]):
                arms[f"{j}|{label}"] = float(np.mean(q[region, j]))
    return named(arms, scenario)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    law = SCENARIO_LAWS[scenario]
    frame, truth = drawn
    result = fit_cleverly(frame, scenario)
    data = result.data
    realised = design(result)
    stratified = scenario == "strata"
    check = named(longhand(result, realised, stratified=stratified), scenario)
    shipped = (
        named(longhand(result, shipped_design(result, scenario), stratified=stratified), scenario)
        if scenario in SHIPPED_SCENARIOS
        else {}
    )
    sample = frame.loc[:, ["W1", "W2", "W3"]].copy()
    sample["S"] = frame["S"].to_numpy() if law.stratified else 0
    treatment = np.asarray(data.treatment, dtype=float)
    position = {code: j for j, code in enumerate(data.arm_codes)}
    sample["A"] = [position[code] if np.isfinite(code) else np.nan for code in treatment]
    sample["dA"] = np.asarray(data.treatment_recorded, dtype=float)
    sample["dY"] = np.asarray(data.observed, dtype=float)
    sample["Y"] = frame["Y"].to_numpy()
    for j in range(3):
        if j < len(data.arm_codes):
            code = data.arm_codes[j]
            sample[f"q{j}"] = np.asarray(result.nuisance.outcome.arms[code], dtype=float)
            sample[f"g{j}"] = realised[:, j]
        else:
            sample[f"q{j}"] = np.nan
            sample[f"g{j}"] = np.nan
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    rows = primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=STUDY.scenarios[scenario],
        initials=initials(result, scenario),
    )
    at_bound = rows_at_bound(result)
    for row in rows:
        name = row["estimand"]
        row["shipped_estimate"] = float(shipped[name]) if shipped else np.nan
        row["longhand_gap"] = abs(float(check[name]) - float(row["estimate"]))
        row["rows_at_bound"] = at_bound
    return sample, truth_rows, rows


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    # No reference input: the reference reads the fit's nuisances (each sample's q and g).
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for scenario in SCENARIO_LAWS for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes,
        estimates.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]],
        samples=samples,
        truths=truths,
    )


#: The columns the Python rows carry beside the published schema, read by
#: :func:`extra_artifacts`.
EXTRA_COLUMNS = ("shipped_estimate", "longhand_gap", "rows_at_bound")
#: The column the R rows carry beside the published schema.
REFERENCE_COLUMNS = ("gn_max_abs_diff",)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The reported design gaps and RMSE ratios, and the per-replicate fit diagnostics.

    ``design-gap.csv`` publishes, per scenario of :data:`SHIPPED_SCENARIOS` and estimand,
    ``mean |psi - psi_shipped| / SD`` over the primary replications (``mean_design_gap``),
    where ``SD`` is the empirical standard deviation of the subject's estimate, and the RMSE
    of the subject and of the shipped construction against the truth with their ratio
    (``rmse_ratio``).  On the ``strata`` scenario this is ``design_gap_strata``, on
    ``missing_outcome`` ``design_gap_missing``.  None is a verdict.

    ``fit-diagnostics.csv`` publishes, per scenario and replication, the rows at a factor
    bound, the largest gap between R ``drtmle``'s product and this package's, and the
    longhand's distance to the fit.
    """
    subject = rows.loc[rows["implementation"] == IMPLEMENTATION].copy()
    shipped = subject.loc[subject["scenario"].isin(SHIPPED_SCENARIOS)].copy()
    records = []
    for (scenario, estimand), group in shipped.groupby(["scenario", "estimand"], sort=True):
        estimate = group["estimate"].to_numpy(dtype=float)
        other = group["shipped_estimate"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        spread = float(np.std(estimate, ddof=1))
        gap = np.abs(estimate - other) / spread
        rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
        rmse_shipped = float(np.sqrt(np.mean((other - truth) ** 2)))
        records.append(
            {
                "scenario": scenario,
                "estimand": estimand,
                "replicates": len(group),
                "empirical_sd": spread,
                "mean_design_gap": float(np.mean(gap)),
                "max_design_gap": float(np.max(gap)),
                "rmse": rmse,
                "rmse_shipped": rmse_shipped,
                "rmse_ratio": rmse / rmse_shipped,
            }
        )
    gap_frame = pd.DataFrame.from_records(records)
    reference = rows.loc[rows["implementation"] == REFERENCE]
    keys = ["scenario", "replicate"]
    bound = subject.groupby(keys)["rows_at_bound"].first()
    check = subject.groupby(keys)["longhand_gap"].max()
    mechanism = reference.groupby(keys)["gn_max_abs_diff"].max()
    diagnostics = pd.concat([bound, check, mechanism], axis=1).reset_index()
    return {"design-gap.csv": gap_frame, "fit-diagnostics.csv": diagnostics}


#: The largest mean paired gap :func:`paired_gaps` may read, in empirical SDs of the subject.
#: Added after the run.  The paired margin of 0.15 SD cannot tell this study's design from the
#: shipped one, whose ``mean_design_gap`` in ``design-gap.csv`` is 0.024 or more on every row.
#: The measured largest paired gap is 0.0008 (``strata``, ``ey0[S=2]``).
PAIRED_GAP_CEILING = 0.005


def paired_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    """The mean absolute gap between the two implementations, per paired row.

    Parameters
    ----------
    rows : DataFrame
        ``replicates.parquet``.

    Returns
    -------
    DataFrame
        One row per scenario and estimand: ``mean |psi_cleverly - psi_drtmle| / SD`` over the
        replications both implementations fitted, as ``mean_paired_gap``, where ``SD`` is the
        empirical standard deviation of the subject's estimate, the scale of
        ``design-gap.csv``.
    """
    wide = rows.pivot_table(
        index=["scenario", "estimand", "replicate"],
        columns="implementation",
        values="estimate",
    ).dropna(subset=[IMPLEMENTATION, REFERENCE])
    records = []
    for (scenario, estimand), group in wide.groupby(level=[0, 1], sort=True):
        subject = group[IMPLEMENTATION].to_numpy(dtype=float)
        reference = group[REFERENCE].to_numpy(dtype=float)
        spread = float(np.std(subject, ddof=1))
        records.append(
            {
                "scenario": scenario,
                "estimand": estimand,
                "replicates": len(group),
                "mean_paired_gap": float(np.mean(np.abs(subject - reference))) / spread,
            }
        )
    return pd.DataFrame.from_records(records)


# ------------------------------------------------------------------------- the reading rule


def routed_truth_row(row: Any) -> bool:
    """Whether a red truth row meets the declared F19 route.

    Red only on its coverage or SE-ratio reading: ``bias_equivalent`` is true, the point SE
    ratio is at least :data:`ROUTED_SE_RATIO_FLOOR` and below 1, and the upper end of its 99%
    SE-ratio interval is below :data:`ROUTED_SE_RATIO_CEILING`.
    """
    return bool(
        row["implementation"] in (IMPLEMENTATION, REFERENCE)
        and bool(row["bias_equivalent"])
        and ROUTED_SE_RATIO_FLOOR <= float(row["se_ratio"]) < 1.0
        and float(row["se_ratio_ci_upper"]) < ROUTED_SE_RATIO_CEILING
    )


def routed_ladder_row(row: Any, se_ratio_upper: float) -> bool:
    """Whether a red ``root_n_and_efficiency`` rung meets the declared F19 route.

    Parameters
    ----------
    row : Any
        The rung's row of ``properties.csv``.
    se_ratio_upper : float
        The upper end of the rung's 99% bootstrap SE-ratio interval.

    Returns
    -------
    bool
        Whether the rung is red only on its coverage or SE-ratio reading, with a point SE
        ratio of at least :data:`ROUTED_SE_RATIO_FLOOR` and below 1.
    """
    return bool(
        bool(row["bias_equivalent"])
        and ROUTED_SE_RATIO_FLOOR <= float(row["se_ratio"]) < 1.0
        and se_ratio_upper < ROUTED_SE_RATIO_CEILING
    )


def defect_suspects(
    performance: pd.DataFrame,
    equivalence: pd.DataFrame,
    properties: pd.DataFrame,
    ladder_se_ratio_upper: Mapping[str, float],
) -> list[str]:
    """Every committed row the reading rule calls a defect suspect.

    Parameters
    ----------
    performance : DataFrame
        ``performance-tests.csv``.
    equivalence : DataFrame
        ``equivalence.csv``.
    properties : DataFrame
        ``properties.csv``.
    ladder_se_ratio_upper : Mapping
        Each ladder rung's cell to the upper end of its 99% bootstrap SE-ratio interval.

    Returns
    -------
    list of str
        One sentence per suspect.  Empty when every red row is routed to F19.
    """
    out: list[str] = []
    for _, row in performance.loc[~performance["passed"].astype(bool)].iterrows():
        key = f"{row['implementation']}/{row['scenario']}/{row['estimand']}"
        if not routed_truth_row(row):
            out.append(f"truth row {key} is red outside the F19 route")
    for _, row in equivalence.loc[~equivalence["passed"].astype(bool)].iterrows():
        out.append(f"paired row {row['scenario']}/{row['estimand']} is red")
    red = ~properties["passed"].astype(bool) | ~properties["property_passed"].astype(bool)
    for _, row in properties.loc[red].iterrows():
        key = f"{row['property']}/{row['cell']}"
        if row["property"] == "root_n_and_efficiency" and routed_ladder_row(
            row, float(ladder_se_ratio_upper[row["cell"]])
        ):
            continue
        out.append(f"property row {key} is red")
    failed = properties.loc[properties["failed_replicates"].fillna(0).astype(int) > 0]
    for _, row in failed.iterrows():
        out.append(f"property row {row['property']}/{row['cell']} lost a replication")
    return out


def routed_rows(
    performance: pd.DataFrame,
    properties: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """The red truth and ladder rows the reading rule routes to F19, as ledger keys.

    Returns
    -------
    tuple of list of str
        ``implementation/scenario/estimand`` for each routed truth row, and
        ``property/cell`` for each red ladder rung.  The caller adds them to
        ``red_cells.CLAIMS["F19"]`` once :func:`defect_suspects` is empty.
    """
    truth = [
        f"{row['implementation']}/{row['scenario']}/{row['estimand']}"
        for _, row in performance.loc[~performance["passed"].astype(bool)].iterrows()
    ]
    red = ~properties["passed"].astype(bool) | ~properties["property_passed"].astype(bool)
    ladder = [
        f"{row['property']}/{row['cell']}"
        for _, row in properties.loc[red].iterrows()
        if row["property"] == "root_n_and_efficiency"
    ]
    return truth, ladder
