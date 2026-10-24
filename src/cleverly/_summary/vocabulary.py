"""Every word a result summary shares with another summary.

A summary is printed by several modules, and two modules that spell one fact two ways
teach a reader that they are two facts. Each label, column name, method name, learner
name, note code and evidence phrase that more than one summary can print lives here once.
A seam that needs a new shared word adds it here, with its test, in its own change.

Nothing here is public. Users meet these strings in printed summaries only.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

# ----------------------------------------------------------------- header labels

QUESTION: Final = "Question"
OUTCOME: Final = "Outcome"
ADJUSTED_FOR: Final = "Adjusted for"
METHOD: Final = "Method"
ROWS: Final = "Rows"
TREATED_SHARE: Final = "Treated share"
ARM_SHARES: Final = "Arm shares"
DOSE: Final = "Dose"
OUT_OF_FOLD: Final = "Out-of-fold predictions"
OUTCOME_MODEL: Final = "Outcome model"
TREATMENT_MODEL: Final = "Treatment model"
PROBABILITY_LIMITS: Final = "Probability limits"
MISSINGNESS_MODEL: Final = "Missingness model"
#: Appended to a share label on weighted data, whose shares the weights tilt.
WEIGHTED: Final = " (weighted)"
#: The model of whether a row's treatment was recorded.
RECORDING_MODEL: Final = "Recording model"
#: The model of the intermediate variable a controlled direct effect holds fixed.
INTERMEDIATE_MODEL: Final = "Intermediate model"
CLUSTERS: Final = "Clusters"
FEWEST_IN_STRATUM: Final = "Fewest in a stratum"
WEIGHTS: Final = "Weights"
#: Foreseen for the longitudinal and sensitivity summaries.
TIME_POINTS: Final = "Time points"
HORIZON: Final = "Horizon"
CENSORING_MODEL: Final = "Censoring model"
ANALYSIS: Final = "Analysis"
#: Checks (T3): whether a check passed, in the words of :data:`CHECK_RESULTS`.
RESULT: Final = "Result"
#: Refutation tests (T3).
ESTIMATE_TESTED: Final = "Estimate tested"
TESTS_RUN: Final = "Tests run"
REPEATS_PER_TEST: Final = "Repeats per test"

#: Every header label above, so a test can check a summary uses only shared labels.
HEADER_LABELS: Final[frozenset[str]] = frozenset(
    {
        QUESTION,
        OUTCOME,
        ADJUSTED_FOR,
        METHOD,
        ROWS,
        TREATED_SHARE,
        ARM_SHARES,
        DOSE,
        OUT_OF_FOLD,
        OUTCOME_MODEL,
        TREATMENT_MODEL,
        PROBABILITY_LIMITS,
        MISSINGNESS_MODEL,
        CLUSTERS,
        FEWEST_IN_STRATUM,
        WEIGHTS,
        TIME_POINTS,
        HORIZON,
        CENSORING_MODEL,
        ANALYSIS,
        RESULT,
        ESTIMATE_TESTED,
        TESTS_RUN,
        REPEATS_PER_TEST,
    }
)

# ----------------------------------------------------------------- column labels

ESTIMATE: Final = "estimate"
P_VALUE: Final = "p-value"
DF: Final = "df"
SUPPORT: Final = "support"
EXPERIMENTAL: Final = "experimental"
MEANING: Final = "meaning"
BAND: Final = "band"
#: The suffix a row label takes when its estimate supplies no inference in a table whose
#: other rows do (docs/architecture-invariants.md, inference status).
NO_INFERENCE_MARK: Final = "*"
#: The cell a table prints where a value does not apply.
MISSING: Final = "-"


def interval_label(level: float) -> str:
    """The interval column label for a confidence level such as ``0.95``.

    Parameters
    ----------
    level : float
        The confidence level, between 0 and 1.

    Returns
    -------
    str
        For example ``"95% CI"``.
    """
    return f"{level * 100:g}% CI"


#: Plain column labels for the spread names :func:`~cleverly.inference.influence.spread_name`,
#: ``spread_columns()`` and :func:`~cleverly.inference.influence.bootstrap_column` publish.
#: A diagnostic name says so in its label, so no diagnostic reads as a standard error.
SPREAD_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "std_err": "std. error",
        "plugin_std_err": "plug-in SE (diagnostic)",
        "ci_lower": "CI lower",
        "ci_upper": "CI upper",
        "plugin_interval_lower": "plug-in range lower (diagnostic)",
        "plugin_interval_upper": "plug-in range upper (diagnostic)",
        "plugin_interval": "plug-in range (diagnostic)",
        "p_value": "p-value",
        "plugin_p_value": "plug-in p-value (diagnostic)",
        "bootstrap se": "bootstrap SE",
        "bootstrap sd": "bootstrap SD (diagnostic)",
        "bootstrap_se": "bootstrap SE",
        "bootstrap_sd": "bootstrap SD (diagnostic)",
        "bootstrap_ci_lower": "bootstrap CI lower",
        "bootstrap_ci_upper": "bootstrap CI upper",
        "bootstrap_range_lower": "bootstrap range lower (diagnostic)",
        "bootstrap_range_upper": "bootstrap range upper (diagnostic)",
    }
)

# ----------------------------------------------------------------- methods

#: The plain name of each fitted method, keyed by ``fitted_method``.
METHOD_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "targeted maximum likelihood (TMLE)",
        "collaborative_tmle": "collaborative TMLE (C-TMLE)",
        "drtmle": "doubly robust TMLE (DR-TMLE)",
        "adaptive_iptw": "adaptive-propensity inverse probability weighting",
    }
)

#: The short name a summary title carries, keyed by ``fitted_method``.
METHOD_SHORT_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "TMLE",
        "collaborative_tmle": "C-TMLE",
        "drtmle": "DR-TMLE",
        "adaptive_iptw": "adaptive IPTW",
    }
)


def method_name(fitted_method: str) -> str:
    """The plain method name, with TMLE for a method this table does not list yet."""
    return METHOD_NAMES.get(fitted_method, METHOD_NAMES["tmle"])


def method_short_name(fitted_method: str) -> str:
    """The short method name, with TMLE for a method this table does not list yet."""
    return METHOD_SHORT_NAMES.get(fitted_method, METHOD_SHORT_NAMES["tmle"])


#: The title of a point-treatment estimate summary, filled with the short method name.
ESTIMATE_TITLE: Final = "Effect estimate ({method})"

# ----------------------------------------------------------------- learners

#: Plain learner names, keyed by class name.
LEARNER_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "LinearRegression": "linear regression",
        "LogisticRegression": "logistic regression",
        "LogisticRegressionCV": "logistic regression (tuned penalty)",
        "Ridge": "ridge regression",
        "RidgeCV": "ridge regression (tuned penalty)",
        "Lasso": "lasso",
        "LassoCV": "lasso",
        "ElasticNet": "elastic net",
        "ElasticNetCV": "elastic net",
        "PoissonRegressor": "Poisson regression",
        "GammaRegressor": "gamma regression",
        "TweedieRegressor": "Tweedie regression",
        "GLM": "generalized linear model",
        "ParametricOutcome": "parametric logistic model",
        "ParametricMechanism": "parametric logistic model",
        "L1Logistic": "lasso logistic regression",
        "RandomForestRegressor": "random forest",
        "RandomForestClassifier": "random forest",
        "ExtraTreesRegressor": "extra trees",
        "ExtraTreesClassifier": "extra trees",
        "HistGradientBoostingRegressor": "gradient boosting",
        "HistGradientBoostingClassifier": "gradient boosting",
        "GradientBoostingRegressor": "gradient boosting",
        "GradientBoostingClassifier": "gradient boosting",
        "KNeighborsRegressor": "nearest neighbours",
        "KNeighborsClassifier": "nearest neighbours",
        "DecisionTreeRegressor": "decision tree",
        "DecisionTreeClassifier": "decision tree",
        "StandardScaler": "scaling",
        "PolynomialFeatures": "polynomial terms",
        "SplineTransformer": "splines",
        "OneHotEncoder": "one-hot coding",
        "MLPRegressor": "neural network",
        "MLPClassifier": "neural network",
    }
)

#: The name of the learner a fit uses when the caller passes none.
DEFAULT_LEARNER: Final = "super learner (default library)"
#: The name of an ensemble, filled with its candidate count.
SUPER_LEARNER: Final = "super learner ({k} candidates)"
#: The name of a model the data declares known rather than one the fit estimated.
KNOWN_MECHANISM: Final = "known (declared)"

#: The learner classes whose in-sample fit cannot make an interval too narrow through
#: overfitting: fixed-dimension parametric models, matched by exact class name. Seeded from
#: ``PER_ARM_PARAMETRIC_OPTIONS`` (``ParametricOutcome``, ``ParametricMechanism``), the only
#: classification the package had. Lasso-type learners select terms from the data and are
#: left out; so is every class not named here, which counts as flexible.
PARAMETRIC_LEARNERS: Final[frozenset[str]] = frozenset(
    {
        "ParametricOutcome",
        "ParametricMechanism",
        "GLM",
        "LinearRegression",
        "LogisticRegression",
        "Ridge",
        "PoissonRegressor",
        "GammaRegressor",
        "TweedieRegressor",
    }
)

#: Pipeline steps that leave a parametric final step parametric.
PARAMETRIC_STEPS: Final[frozenset[str]] = frozenset({"StandardScaler"})

# ----------------------------------------------------------------- estimands

#: What one row of a point-treatment table estimates, keyed by target name. ``{Y}`` is the
#: outcome, ``{A}`` the treatment, ``{v}`` the row's arm and ``{r}`` the reference arm.
ESTIMAND_MEANINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ate": (
            "average effect: mean {Y} if every unit had {A} = {v}, minus the mean if every "
            "unit had {A} = {r}"
        ),
        "att": (
            "average effect among units that had {A} = {v}: their mean {Y} with {A} = {v}, "
            "minus their mean if they had had {A} = {r}"
        ),
        "atc": (
            "average effect among units that had {A} = {r}: their mean {Y} if they had had "
            "{A} = {v}, minus their mean with {A} = {r}"
        ),
        "ey": "mean {Y} if every unit had {A} = {v}",
        "ey1": "mean {Y} if every unit had {A} = {v}",
        "ey0": "mean {Y} if every unit had {A} = {v}",
        "ey_obs": "mean {Y} with each unit's treatment as received",
        "rr": (
            "risk ratio: mean {Y} if every unit had {A} = {v}, divided by the mean if every "
            "unit had {A} = {r}"
        ),
        "or": (
            "odds ratio: odds of {Y} if every unit had {A} = {v}, divided by the odds if "
            "every unit had {A} = {r}"
        ),
        "par": (
            "mean {Y} with each unit's treatment as received, minus the mean if every unit "
            "had {A} = {r}"
        ),
        "paf": (
            "share of the mean {Y} with treatment as received that would go away if every "
            "unit had {A} = {r}"
        ),
    }
)

#: The Question fact of a single-row table: the meaning's opening words, keyed by target.
QUESTION_PHRASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ate": "average effect of {A} on {Y} ({v} compared with {r})",
        "att": "average effect of {A} on {Y} among units with {A} = {v} ({v} compared with {r})",
        "atc": "average effect of {A} on {Y} among units with {A} = {r} ({v} compared with {r})",
        "ey": "mean {Y} if every unit had {A} = {v}",
        "ey1": "mean {Y} if every unit had {A} = {v}",
        "ey0": "mean {Y} if every unit had {A} = {v}",
        "ey_obs": "mean {Y} with treatment as received",
        "rr": "risk ratio of {Y} for {A} ({v} compared with {r})",
        "or": "odds ratio of {Y} for {A} ({v} compared with {r})",
        "par": "mean {Y} as treated, minus the mean if every unit had {A} = {r}",
        "paf": "share of mean {Y} that would go if every unit had {A} = {r}",
    }
)

#: The Question fact when a table reports several rows; the key table says what each is.
SEVERAL_ROWS_QUESTION: Final = "effect of {A} on {Y}; each row is defined below"
#: The Question fact of a controlled direct effect.
DIRECT_EFFECT_QUESTION: Final = "effect of {A} on {Y} with {Z} held at {z} for every unit"
#: The Question fact of a learned-rule fit, whose target the data chose.
LEARNED_RULE_QUESTION: Final = "value of a treatment rule learned from the data"
#: The title of the key table.
KEY_TITLE: Final = "What each row estimates"
#: A per-stratum row: the row's meaning restricted to the units of one baseline stratum.
STRATUM_MEANING: Final = "{meaning}, among units with {stratum}"

# ----------------------------------------------------------------- notes

NoteKind = Literal["warning", "note"]

#: Every note code, with its kind. A ``warning`` changes how to read a number; a ``note``
#: explains one. A summary may only print a code listed here.
NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "no_inference": "warning",
        "experimental": "note",
        "in_sample_flexible": "warning",
        "weights_capped": "note",
        "weighted_population": "note",
        "t_reference": "note",
        "spread_scale": "note",
        "repeated_splits": "note",
        "simultaneous_bands": "note",
        "bootstrap_diagnostic": "note",
        "bootstrap_experimental": "note",
        "score_unsolved": "warning",
        "stopping_rule": "note",
        "folds_reduced": "note",
        "fit_warning_positivity": "warning",
        "fit_warning_convergence": "warning",
        "fit_warning_weighting": "warning",
        "fit_warning_sparse_influence": "warning",
        "fit_warning_monte_carlo": "warning",
        "fit_warning_parameter_space": "warning",
        "fit_warning_data": "warning",
        "fit_warning_other": "warning",
        # Refutation tests (T3): one note per test, a warning when that test failed.
        "test_placebo": "note",
        "test_random_common_cause": "note",
        "test_subset": "note",
        "test_negative_control_outcome": "note",
        "test_dummy_outcome": "note",
        "test_simulated_outcome": "note",
        "test_bootstrap_measurement_error": "note",
        "test_bootstrap_stability": "note",
        "test_plasmode": "note",
        "test_future_treatment": "note",
        "refit_failures": "warning",
        "refit_spread_diagnostic": "warning",
        "ratio_scale": "note",
    }
)

#: How a cross-fitted fit built its estimate, from its targeting scheme and evaluation.
CROSS_FIT_CONSTRUCTION: Final = (
    "The models predict each row from the folds that leave it out; {targeting}, and {evaluation}."
)
#: Keyed by whether one targeting step serves every fold (``targeting_scheme="pooled"``).
CROSS_FIT_TARGETING: Final[Mapping[bool, str]] = MappingProxyType(
    {
        True: "one final adjustment of the outcome predictions serves every fold",
        False: "each fold gets its own final adjustment of the outcome predictions",
    }
)
#: Keyed by ``cv_evaluation``.
CROSS_FIT_EVALUATION: Final[Mapping[bool, str]] = MappingProxyType(
    {
        False: "each estimate is computed once on all rows",
        True: "each estimate is the average of its estimates in the held-out folds",
    }
)

#: What a score-check row that belongs to no reported row covers.
SCORE_MODEL_ROW: Final = "the fitted models"

#: Plain names of the axes a support reason says the nearest validated fit differs on, keyed
#: by the names in ``cleverly._support.KEY_AXES``.
SUPPORT_AXES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "kind": "the kind of estimator",
        "engine": "the estimation method",
        "family": "the estimands",
        "outcome": "the outcome type or its declared range",
        "missing": "missing data",
        "treatment": "the treatment type",
        "weighting": "observation weights",
        "splitting": "how the sample is split for the models",
        "dependence": "clustering or the number of clusters",
        "qualifiers": "strata or a known treatment mechanism",
        "options": "estimator options",
    }
)

#: Fit-time warnings a note already states, by that note's code: when the note prints, a
#: recorded warning whose message holds one of these phrases is left out of the Warnings
#: block, and the note prints among the warnings in its place. The phrases are filled with
#: the fit's folds (``declared``, ``used``) and mean treatment bounds (``low``, ``high``),
#: so a warning about other folds or other bounds stays.
RESTATED_WARNINGS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "folds_reduced": ("reducing n_folds from {declared} to", "to {used}: only"),
        "weights_capped": (
            "treatment probability outside the truncation bounds [{low}, {high}]",
            "values fall below the nuisance bound",
        ),
    }
)

#: The note code of a recorded fit-time warning, keyed by the warning class name.
WARNING_NOTE_CODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PositivityWarning": "fit_warning_positivity",
        "ConvergenceWarning": "fit_warning_convergence",
        "WeightingWarning": "fit_warning_weighting",
        "SparseInfluenceWarning": "fit_warning_sparse_influence",
        "MonteCarloWarning": "fit_warning_monte_carlo",
        "ParameterSpaceWarning": "fit_warning_parameter_space",
        "DataWarning": "fit_warning_data",
    }
)

#: Why a row is experimental, by the kind of gap its support reason names
#: (``_estimates.reason_kind``). The full summary prints the reason itself under Settings.
EXPERIMENTAL_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "no_row": (
            "No validation study covers this kind of fit, so cleverly marks {names} experimental."
        ),
        "family": (
            "The validation study of this kind of fit covers other outcome types or another "
            "kind of interval, so cleverly marks {names} experimental."
        ),
        "condition": (
            "This fit has a feature the validation study of its kind does not cover, so "
            "cleverly marks {names} experimental."
        ),
    }
)

#: Why a bootstrap range is a diagnostic, by its source (``bootstrap_is_diagnostic``).
BOOTSTRAP_DIAGNOSTIC: Final[Mapping[str, str]] = MappingProxyType(
    {
        "withheld": (
            "The bootstrap range of {names} is a diagnostic, not a confidence interval, as "
            "the fit withholds that interval."
        ),
        "below_floor": (
            "The bootstrap range of {names} is a diagnostic, not a confidence interval, "
            "because the fit reads fewer than {clusters} clusters, and no validation study "
            "measures a bootstrap interval at so few."
        ),
        "declared": (
            "The bootstrap range of {names} is a diagnostic, not a confidence interval: the "
            "collaborative fit builds its treatment model from the data again on each "
            "resample, and no result shows that this range has its stated coverage."
        ),
    }
)

#: Appended to a quoted warning the fit raised more than once.
REPEATS: Final = " ({count} times)"

NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "refutation_test": (
            "{name}: {purpose}{conditions} For this kind of fit the test {evidence}."
        ),
        "refit_failures": (
            "{failed} of {total} repeats of the {name} test failed and are left out of its "
            'row; report.draws_frame("{test}") lists why.'
        ),
        "refit_spread_diagnostic": (
            "The refits of {names} have no valid standard error, only a plug-in diagnostic, "
            "so their statistics read the {label}."
        ),
        "ratio_scale": "For a ratio, {parts}.",
        "no_inference": (
            "No confidence interval or p-value for {names}. {reason} The {label} column is "
            "a diagnostic, not a standard error."
        ),
        "no_inference_row": (
            "No confidence interval or p-value for {names}, marked *. {reason} Their "
            "{label}: {spreads}."
        ),
        "experimental": (
            "The numbers are computed as for a validated fit; read {what} with more caution. "
            "The full summary gives the reason."
        ),
        "in_sample_flexible": (
            "The models were fitted and evaluated on the same rows. With flexible learners "
            "this can make the interval too narrow; cross-fitting (cross_fit=True, or "
            "CrossFitting() in a study method) gives out-of-fold predictions."
        ),
        "weights_capped": (
            "cleverly limited the estimated treatment probabilities to [{low}, {high}] for "
            "{count} of {n} rows{split} ({reads})."
        ),
        "weights_capped_missingness": (
            "cleverly raised {share} of the estimated probabilities {what}, counted over "
            "every arm, to {bound}."
        ),
        "weights_capped_effect": (
            "Each limit caps the weight of the rows it moves. That steadies the estimate, "
            "and it can move the estimate if the true probabilities lie outside the limits. "
            "result.diagnostics.truncation_curve() shows the estimate under other limits."
        ),
        "weighted_population": (
            "The estimate describes the population that the observation weights represent, "
            "not the sample as drawn. result.data.weight_report() describes the weights."
        ),
        "t_reference": (
            "A row based on fewer than {threshold} clusters with positive weight mass, in "
            "the fit or in its stratum, takes its interval and p-value from a t distribution "
            "with the degrees of freedom in the df column, which widens them; a row marked "
            "normal uses the normal distribution. The full summary gives the rule and its "
            "sources."
        ),
        "spread_scale": (
            "The std. error of {names} is on the log scale, where the interval is computed "
            "before it is turned back to the scale of the estimate."
        ),
        "spread_scale_transform": (
            "The std. error of {names} is on the transformed scale, where the interval is "
            "computed before it is turned back to the scale of the estimate."
        ),
        "repeated_splits": (
            "Each row is the median over {repeats} random splits into folds, and its "
            "{spread} includes the spread between splits. Rows are combined one at a time, "
            "so a difference of two rows need not equal a third."
        ),
        "simultaneous_bands": (
            "The bands cover all {count} rows at once with {level} probability; use them "
            "when you read several rows together."
        ),
        "score_unsolved": (
            "The fit did not solve its estimating equation for {names} within tolerance, so "
            "the {spread} may be wrong. result.diagnostics.score_equations() gives the "
            "numbers."
        ),
        "score_rule_unmet": (
            "The fit did not meet its declared stopping rule for {names}, so the {spread} "
            "may be off by more than the rule allows. result.diagnostics.score_equations() "
            "gives the numbers."
        ),
        "stopping_rule_plain": (
            "The fit stopped at its declared stopping rule, which leaves the estimating "
            "equation solved up to the remainder the rule allows."
        ),
        "stopping_rule": (
            "Under the declared stopping rule the estimating equation is solved up to a "
            "remainder worth {b} standard errors of bias (on {name}). If the standard error "
            "is right, a {level} interval with that bias still covers the truth at least "
            "{bound} of the time."
        ),
        "bootstrap_experimental": (
            "No validation study covers the bootstrap interval of {names} for this kind of "
            "fit, so cleverly marks it experimental."
        ),
        "bands_experimental": (
            "No validation study covers the bands for this kind of fit, so cleverly marks "
            "them experimental."
        ),
        "bands_leave_out": "The bands leave out {names}, which have no interval.",
        "folds_reduced": (
            "The fit asked for {declared} folds and used {used}, because the rarest group "
            "or the cluster count allowed no more."
        ),
        "fit_warning": "During the fit cleverly warned: {messages}",
        "fit_warning_more": (
            "It also gave {count} more warnings of this kind; result.fit_warnings lists them."
        ),
    }
)

# ----------------------------------------------------------------- full detail

SECTION_ASSUMPTIONS: Final = "Assumptions"
SECTION_SETTINGS: Final = "Settings"
SECTION_METHOD: Final = "Method details"
SECTION_STUDY: Final = "Study description"
ASSUMPTIONS_LEAD: Final = "The estimate equals {functional} when these hold:"
ASSUMPTIONS_LEAD_GENERIC: Final = "The estimate has its causal meaning when these hold:"
#: The conditions a fit with no study adds for a partly recorded outcome.
MISSING_OUTCOME_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "outcome missing at random: whether {Y} is recorded is independent of {Y} given {A} and W",
    "recording positivity: P({Y} recorded | {A}, W) > 0 almost surely",
)
#: The conditions a fit with no study adds for a partly recorded treatment.
MISSING_TREATMENT_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "treatment missing at random: the potential outcome is independent of whether {A} is "
    "recorded given {A} and W",
    "recording positivity: P({A} = a and {A} recorded | W) > 0 almost surely for every arm a",
)
#: The conditions a fit with no study adds when it holds an intermediate at one value.
INTERMEDIATE_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "no unmeasured intermediate confounding: {Y}(a, {z}) is independent of {Z} given {A} and W",
    "intermediate positivity: P({Z} = {z} | {A} = a, W) > 0 almost surely for every arm a "
    "(for a continuous {Z}, a positive density at {z})",
)
SENSITIVITY_POINTER: Final = (
    "result.sensitivity.omitted_confounding() shows how strong an unmeasured cause would "
    "need to be to change the conclusion."
)
NO_PROTOCOL: Final = "No study protocol was recorded."

# ----------------------------------------------------------------- internals

#: Words that name the package's internals rather than the analysis. No summary prints
#: them, at either detail (Eric, 2026-10-24: "dont include library arcana like hashes etc
#: - just the parts relevant to the analysis"). The contract test reads this list.
INTERNAL_TERMS: Final[tuple[str, ...]] = (
    "fluctuation",
    "iterative",
    "pooled",
    "fingerprint",
    "schema",
    "nuisance",
    "registry",
    "rung",
    "cache",
    "targeting",
    "submodel",
    "Levy",
    "run_id",
    "docs/roadmap.md",
)

# ----------------------------------------------------------------- checks (T3)

#: What a check found, in the words every check summary prints for it: in the ``Result``
#: header fact and in a table's ``result`` column. The keys are what the code decides.
CHECK_RESULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "passed": "passed",
        "failed": "failed",
        "descriptive": "no pass or fail",
        "warning": "needs attention",
    }
)

#: Plain evidence words for each refutation evidence level (X89 plan 6.6), as a verb phrase
#: that completes "For this kind of fit the test ...". A pass or fail needs the third or
#: fourth level (``validation/refute.py``).
EVIDENCE_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "L0": "gives a description only, with no pass or fail",
        "L1": (
            "compares the refits with a stated reference, but how often it fails a correct "
            "fit is not measured, so it gives no pass or fail"
        ),
        "L2": (
            "gives a pass or fail backed by an exact reference or by a simulation study of "
            "this kind of fit"
        ),
        "L3": (
            "gives a pass or fail backed the same way, with an external package's result beside it"
        ),
        "withheld": "is studied, but gives no pass or fail here because {reason}",
        "studied": "is studied, but gives no pass or fail on this report",
        "not_run": "gives no pass or fail, because no refit ran",
    }
)

#: Each refutation test by what it does, for the ``test`` column.
REFUTATION_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "placebo": "shuffled treatment",
        "random_common_cause": "added noise column",
        "subset": "random subsamples",
        "negative_control_outcome": "negative-control outcome",
        "dummy_outcome": "noise outcome",
        "simulated_outcome": "simulated outcome",
        "bootstrap_measurement_error": "measurement error",
        "bootstrap_stability": "bootstrap stability",
        "plasmode": "plasmode outcome",
        "future_treatment": "future treatment",
    }
)

#: What cleverly does in each refutation test, the lead of that test's note. ``{null}`` is
#: the value the test expects with no effect, read from the test itself.
REFUTATION_PURPOSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "placebo": (
            "cleverly replaces the treatment with a shuffled or independently drawn copy, so "
            "the treatment has no effect and the estimate should be about {null}."
        ),
        "random_common_cause": (
            "cleverly adds a column of pure noise to the adjustment set, so the estimate "
            "should barely move."
        ),
        "subset": (
            "cleverly refits on random subsamples of the rows. If the standard error is "
            "right, the refits spread about as much as the expected SD."
        ),
        "negative_control_outcome": (
            "cleverly refits with the control outcome you gave, which you assume the "
            "treatment cannot affect, so the estimate should be about {null}."
        ),
        "dummy_outcome": (
            "cleverly replaces the outcome with random noise unrelated to the treatment and "
            "the covariates, so the true effect is known, and checks {readings} against it."
        ),
        "simulated_outcome": (
            "cleverly replaces the outcome with one simulated from the process you declared, "
            "with a known effect, and checks {readings} against it."
        ),
        "plasmode": (
            "cleverly resamples your covariates, draws treatment from a frozen treatment "
            "model and the outcome from the process you declared, with a known effect, and "
            "checks {readings} against it."
        ),
        "bootstrap_measurement_error": (
            "cleverly resamples the rows, adds the measurement error you declared to "
            "{variables}, and refits."
        ),
        "bootstrap_stability": (
            "cleverly resamples units, keeping their folds, and refits, to show how far the "
            "estimate moves."
        ),
        "placebo_not_run": (
            "cleverly would replace the treatment with a shuffled copy, but runs no refit here."
        ),
        "future_treatment": (
            "cleverly estimates the effect of a later treatment on a control you measured "
            "before it, which the later treatment cannot cause, so the estimate should be "
            "about {null}."
        ),
    }
)

#: Plain names of a refutation statistic, keyed by the test's ``statistic_name`` or the
#: label of one of its extra numbers.
STATISTIC_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "t": "t of the refits against {null}",
        "z": "z of the estimate against {null}",
        "scatter ratio rho": "spread ratio (refit SD / expected SD)",
        "paired shift in standard errors": "shift, in std. errors",
        "paired shift in plug-in standard errors": "shift, in plug-in SEs (diagnostic)",
        "mean reported SE over SD of the draws": "mean std. error / refit SD",
        "mean plug-in SE over SD of the draws": "mean plug-in SE (diagnostic) / refit SD",
        "spread of the paired differences (SE units)": "SD of the shifts, in std. errors",
        "spread across seeds with no noise column (SE units)": (
            "SD across seeds without noise, in std. errors"
        ),
        "shift against the original fit": "shift from the original fit",
        "standardized bias": "standardized bias",
        "bias": "std. bias",
        "se_ratio": "SE ratio",
        "coverage": "coverage",
    }
)

#: The reading sentence of a refutation report, by its result.
REFUTATION_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "passed": "Every test that gives a pass or fail passed.",
        "failed": (
            "{names} failed. A failure points to the kind of defect the test is built to "
            "find, but a correct fit also fails by chance, at about the rate its evidence "
            "measured or its exact reference sets; the note for each test says what it "
            "can show."
        ),
        "descriptive_all": (
            "No test failed, and none of the {n} tests gives a pass or fail for this kind of fit, "
            "so compare each mean with the expected column."
        ),
        "descriptive_one": (
            "The test gives no pass or fail for this kind of fit, so compare its mean with "
            "the expected column."
        ),
        "descriptive_both": (
            "No test failed, and neither test gives a pass or fail for this kind of fit, so "
            "compare each mean with the expected column."
        ),
        "descriptive_some": (
            "No test failed. {k} of the {n} tests {verb} no pass or fail for this kind of "
            "fit, so compare their means with the expected column."
        ),
    }
)

#: The outcome processes a generated-outcome test can declare, in plain words, keyed by the
#: process's ``name``.
PROCESS_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "gaussian_independent_noise": "Gaussian noise unrelated to the data",
        "gaussian_adjustment_dependent": "a Gaussian outcome built from your covariates",
        "bernoulli_adjustment_dependent": "a binary outcome built from your covariates",
        "bounded_beta_adjustment_dependent": "a bounded outcome built from your covariates",
        "logistic_hazard": "event times from a logistic hazard",
        "multinomial_hazard": "competing event times from a multinomial hazard",
    }
)

#: What a generated-outcome test reads, by reading name, for its note.
READING_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"bias": "the bias", "se_ratio": "the std. error", "coverage": "the coverage"}
)

#: The two parts of the ``ratio_scale`` note, each naming the tests it applies to.
RATIO_GEOMETRIC: Final = "the mean of the {names} is the geometric mean of the refits"
RATIO_LOG: Final = "the SD of the {names} is on the log scale"

#: How a resampling test drew its rows, by the report's ``resampling``.
RESAMPLING_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"iid": "rows resampled one at a time", "cluster": "whole clusters resampled"}
)

#: Section titles of a refutation report's full detail.
SECTION_COMPARED: Final = "What each test was compared with"
SECTION_DRAWS: Final = "Repeats"
REFUTATION_TITLE: Final = "Refutation tests"
STATISTICS_TITLE: Final = "Test statistics"
READINGS_TITLE: Final = "Readings against their allowed bands"
