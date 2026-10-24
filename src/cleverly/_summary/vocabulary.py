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

#: The sensitivity analyses (X89 T4).
ESTIMATE_ANALYSED: Final = "Estimate"
OUTCOME_STRENGTH: Final = "Outcome-side strength"
TREATMENT_STRENGTH: Final = "Treatment-side strength"
ALIGNMENT: Final = "Alignment"
LARGEST_SHIFT: Final = "Largest shift"
CONCLUSION: Final = "Conclusion"
BENCHMARK_COVARIATES: Final = "Benchmark covariates"
IMPLIED_ALIGNMENT: Final = "Implied alignment"
GRID: Final = "Grid"
SCALE: Final = "Scale"
MISSING_OUTCOMES: Final = "Missing outcomes"
DIRECTION: Final = "Direction"
MOVEMENT: Final = "Movement"
CELLS: Final = "Cells"
POPULATION: Final = "Population"
REPEATED_SPLITS: Final = "Repeated splits"

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
        ESTIMATE_ANALYSED,
        OUTCOME_STRENGTH,
        TREATMENT_STRENGTH,
        ALIGNMENT,
        LARGEST_SHIFT,
        CONCLUSION,
        BENCHMARK_COVARIATES,
        IMPLIED_ALIGNMENT,
        GRID,
        SCALE,
        MISSING_OUTCOMES,
        DIRECTION,
        MOVEMENT,
        CELLS,
        POPULATION,
        REPEATED_SPLITS,
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
        "robustness_value_ci": "limit robustness value",
        "rva": "limit robustness value",
        "robustness_value_plugin_interval": "plug-in limit robustness value (diagnostic)",
        "rv_plugin_interval": "plug-in limit robustness value (diagnostic)",
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
        # The sensitivity analyses (X89 T4).
        "sensitivity_assumption": "note",
        "no_inference_sensitivity": "warning",
        "limits_withheld": "warning",
        "limits_descriptive": "warning",
        "rv_limit": "note",
        "alignment_stated": "note",
        "joint_strength": "note",
        "consistent_mechanism": "note",
        "strength_clipped": "note",
        "grid_thinned": "note",
        "approximate_scale": "note",
        "censoring_conditions": "note",
        "per_arm_conditions": "note",
        "bound_truncated": "note",
        "tilt_diagnostic": "note",
        "association_reading": "note",
        "population_moves": "note",
        "incremental_reading": "note",
        "learned_rule_reading": "note",
        "observation_reads": "note",
        "bounded_rows": "note",
        "redrawn_folds": "note",
        "cells_failed": "warning",
        "calibration_nan": "note",
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
        # The sensitivity analyses (X89 T4). ``assumption_<analysis>`` texts are the
        # ``sensitivity_assumption`` note of each analysis; ``verdict_*`` texts are its
        # one reading sentence.
        "no_inference_sensitivity": (
            "No confidence statement for {names}. {reason} The {label} is a diagnostic, "
            "not a confidence limit."
        ),
        "assumption_bounds": (
            "The range is the estimate moved down and up by the largest bias that variables "
            "the adjustment left out could, at worst, cause at the stated strengths. cf_y is "
            "the share of the outcome variation left after adjustment that they would "
            "explain, cf_d the share of the average squared weight that only they would add, "
            "and rho how far their two effects line up (one is the worst case)."
        ),
        "assumption_bounds_limits": (
            "Each one-sided limit is a {level} confidence limit for its own end of the range: "
            "the lower limit for the lower end, the upper limit for the upper end."
        ),
        "verdict_rv_reached": (
            "Left-out variables with both strengths at {rv} could, at worst, move the "
            "estimate to {null}{alignment}; weaker ones, with both strengths below {rv}, "
            "could not."
        ),
        "verdict_rv_unreached": (
            "No left-out variables with equal strengths below one could move the estimate to "
            "{null}{alignment}."
        ),
        "rv_limit": (
            "The {limit} robustness value (RVa) is the same strength for the {level} "
            "one-sided limit nearest {null}: below it, that limit stays on its side of "
            "{null}."
        ),
        "assumption_robustness": (
            "The robustness value is the strength that left-out variables would need, with "
            "the same value for cf_y (the share of the outcome variation left after "
            "adjustment that they would explain) and cf_d (the share of the average squared "
            "weight that only they would add), to move the estimate to {null}."
        ),
        "alignment_stated": (
            "rho = {rho} assumes the two effects of the left-out variables only partly line "
            "up; one is the worst case."
        ),
        "assumption_elements": (
            "The bias scale is the square root of the product of the first two rows. A bound "
            "multiplies it by a factor that grows with the strengths of the left-out "
            "variables, so these rows give the range at any strengths."
        ),
        "assumption_benchmark": (
            "A common calibration (Chernozhukov et al.) takes these as the strengths that "
            "left-out variables would have if leaving them out hurt the fit as much as "
            "leaving out {covariates} does. Leaving those out moved the estimate by {delta}."
        ),
        "benchmark_pointer": (
            "Pass this record as benchmark= to result.sensitivity.omitted_confounding() for "
            "the range at these strengths."
        ),
        "assumption_contour": (
            "Each cell is the {bound} end of the range for {estimand} when the variables the "
            "adjustment left out have the outcome-side strength of its row (cf_y) and the "
            "treatment-side strength of its column (cf_d), at alignment {rho}."
        ),
        "verdict_evalue": (
            "An unmeasured confounder would need a risk ratio of {point} with both the "
            "treatment and the outcome, beyond the measured covariates, to explain away the "
            "estimate; {limit_sentence}"
        ),
        "evalue_limit_reached": "{limit} to move the interval to the null.",
        "evalue_limit_covers": "the interval already includes the null.",
        "assumption_evalue": (
            "The E-value is the strength at which the confounder's associations with the "
            "treatment and with the outcome, taken as equal, could explain the estimate "
            "away. Unequal strengths can trade off: one association may be below the "
            "E-value if the other is above it."
        ),
        "assumption_evalue_per_arm": (
            "The estimate holds the intermediate fixed, so its bias is a product of one "
            "bounding factor per arm, and the E-value is the equal strength of all four "
            "risk-ratio associations, two per arm. Unequal strengths can trade off."
        ),
        "evalue_conversion_or": (
            "The odds ratio was converted to a risk ratio by its square root, an "
            "approximation for common outcomes that can land above or below the true risk "
            "ratio."
        ),
        "evalue_conversion_rd": (
            "The risk difference was converted to a risk ratio with the estimated risk of "
            "the reference arm, held fixed, so the sampling error of that risk is ignored."
        ),
        "evalue_conversion_smd": (
            "The mean difference was divided by the outcome's standard deviation and "
            "converted to a risk ratio through an odds ratio; each step is an approximation."
        ),
        "evalue_derived": (
            "The risk ratio {name} was derived from the models fitted for the reported "
            "{source}, with no model refit."
        ),
        "censoring_conditions": (
            "The risk is the chance of the event by the horizon. The E-value also assumes "
            "that censoring is random given the treatment and covariates, and that the "
            "unmeasured confounder does not affect censoring."
        ),
        "per_arm_conditions": (
            "It also assumes the confounder is a baseline variable the treatment does not "
            "affect, that every unit could receive each treatment and intermediate level, "
            "and that no common cause of the intermediate and the outcome is affected by "
            "the treatment."
        ),
        "assumption_marginal": (
            "gamma is the largest factor by which an unmeasured confounder may change a "
            "unit's odds of treatment, given the covariates, in either direction, with no "
            "limit on its link to the outcome. The range holds every value of {estimand} "
            "that allows, if this confounder is all the adjustment missed. At a gamma of one, no "
            "unmeasured confounding, the range shrinks to one value, the bound's own doubly "
            "robust estimate, close to the fit's."
        ),
        "marginal_interval": (
            "In large samples the interval contains the true range with probability at "
            "least {level}."
        ),
        "assumption_tilt": (
            "gamma is how far the average outcome of units whose outcome is missing differs "
            "from that of recorded units with the same treatment and covariates, on the "
            "log-odds scale of the outcome rescaled to run from zero to one{arm_clause}. A gamma of zero is the "
            "main analysis, which assumes outcomes are missing at random, and a positive "
            "gamma means the missing outcomes were higher. At every gamma the interval "
            "reuses the main analysis's standard error and treats gamma as known, so it "
            "describes sampling error at that gamma, not doubt about gamma."
        ),
        "assumption_surface": (
            "Each cell adds a simulated common cause of {moved} and the outcome, refits the "
            "analysis, and reports how far the estimate moved. The surface is qualitative: "
            "it is not a bound and carries no interval. {laws}"
        ),
        "law_binary": (
            "The treatment strength is the share of units, those highest on the simulated "
            "cause, whose treatment is flipped."
        ),
        "law_categorical": (
            "The treatment strength is the share of units in arms {a} and {b}, those highest "
            "on the simulated cause, whose arms are swapped."
        ),
        "law_continuous": (
            "The treatment strength is how far the dose moves per standard deviation of the "
            "simulated cause."
        ),
        "law_vector": (
            "Each binary component flips for a share of units, those highest on the "
            "simulated cause, set by its strength, and each dose moves by its strength per "
            "standard deviation of the cause."
        ),
        "law_longitudinal": (
            "Each treatment node flips for a share of units, those highest on the simulated "
            "cause, set by its strength; one cause value per unit is shared by every node "
            "and the outcome."
        ),
        "law_intermediate": (
            "Every cell holds the treatment and flips the intermediate {name} for a share of "
            "units, those highest on the simulated cause, so the cause confounds the "
            "intermediate and the outcome."
        ),
        "law_outcome_gaussian": (
            "The outcome moves by its strength, with the sign reversed, per standard "
            "deviation of the simulated cause."
        ),
        "law_outcome_binomial": (
            "The outcome flips for a share of units, those highest on the simulated cause, "
            "set by the outcome strength."
        ),
        "law_outcome_logit": (
            "The outcome moves on the log-odds scale of its declared bounds, so it stays "
            "inside them, and the outcome strength is not in standard-deviation units: "
            "compare cells across fits by their outcome association."
        ),
        "law_outcome_normal_score": (
            "The outcome moves on a normal-score scale and is mapped back inside its "
            "declared bounds, so the law depends on the sample, and the outcome strength is "
            "not in standard-deviation units: compare cells across fits by their outcome "
            "association."
        ),
        "association_defined": (
            "An association column is the correlation, on these data, between the simulated "
            "cause and the treatment{intermediate} the cell produced."
        ),
        "incremental_reading": (
            "Each cell refits the treatment model, and the shift in treatment odds reads "
            "that refitted model, so a cell moves the intervention along with the "
            "confounding."
        ),
        "learned_rule_reading": (
            "Each cell keeps every unit's learned rule fixed, so it reads confounding of "
            "those fixed rules, not of the procedure that learned them."
        ),
        "observation_reads": (
            "The cells that move the treatment are valid only if whether a value is recorded "
            "depends on {reads} alone, as declared."
        ),
        "bounded_rows": (
            "A share {share} of recorded outcomes sits at a bound and never moves in any cell."
        ),
        "redrawn_folds": (
            "Each refit redraws the cross-fitting folds, so every movement includes fold "
            "noise; the anchor is the original fit."
        ),
        "limits_withheld_learner": (
            "No one-sided limits or limit robustness value: the plug-in estimate of the "
            "average squared weight has a standard error only for a "
            "declared known treatment model or a qualifying scikit-learn LogisticRegression, "
            "and this fit's treatment model does not qualify. {alternative}"
        ),
        "limits_withheld_configuration": (
            "No one-sided limits or limit robustness value: this bound's weight also reads "
            "a second model or a stratum share, and the standard error of the plug-in "
            "estimate covers the treatment model alone. "
            "{alternative}"
        ),
        "limits_withheld_correction": (
            "No one-sided limits or limit robustness value: the plug-in estimate of the "
            "average squared weight needs a correction before it "
            "has a standard error, and the correction does not hold on this fit. "
            "{alternative}"
        ),
        "limits_withheld_consistent": (
            "No one-sided limits or limit robustness value: with "
            "assume_consistent_mechanism=True the bound is a point bound, and no cited "
            "source gives its standard error. {alternative}"
        ),
        "limits_withheld_zero": (
            "No one-sided limits or limit robustness value: the bias scale is zero on this "
            "fit, and a zero scale has no standard error."
        ),
        "limits_descriptive": (
            "No validation study licenses the {what} for this kind of fit, so read it as "
            "descriptive."
        ),
        "joint_strength": (
            "Here the treatment-side strength measures how much the left-out variables add "
            "to the weights of the treatment and of {mechanisms} together."
        ),
        "consistent_mechanism": (
            "This point bound also assumes that the treatment model behind its weights is "
            "correct (assume_consistent_mechanism=True); the package does not check it."
        ),
        "strength_clipped_y": (
            "Leaving out {covariates} raised the outcome variation left by a factor above "
            "two (gain {gain}), so cf_y is one: left-out variables this strong would explain "
            "all of the outcome variation left after adjustment."
        ),
        "strength_clipped_d": (
            "The treatment-side gain is {gain}, at least one, so cf_d is one, where the bound "
            "has no finite value."
        ),
        "grid_thinned": (
            "The table shows {rows} of the {y_size} cf_y values and {columns} of the "
            "{d_size} cf_d values; to_frame() holds every cell."
        ),
        "bound_truncated": (
            "The converted lower limit, {value}, lies below zero, outside the risk-ratio "
            "scale, so the table shows zero."
        ),
        "tilt_diagnostic": (
            "For {names} the spread off a gamma of zero is the main analysis's spread around the "
            "moved estimate{scale}, a diagnostic: no interval is claimed there."
        ),
        "population_moves": (
            "Each cell rebuilds the population of units with {arm_phrase} from its own "
            "treatment, so part of a movement can be a change of population; the population "
            "share column shows the share of units in that group."
        ),
        "cells_failed": (
            "{failed} of {total} cells failed, so their movement is missing. "
            "surface.failures gives each error."
        ),
    }
)

#: Symbols a short sensitivity summary leaves to ``detail="full"`` (X89 plan 8.1 row T4).
SHORT_DETAIL_SYMBOLS: Final[tuple[str, ...]] = ("sigma^2", "nu^2", "Riesz")

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

#: Plain evidence words for each refutation rung (X89 plan 6.6). A verdict needs L2 or
#: above (``validation/refute.py``).
EVIDENCE_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "L0": "description only, no pass or fail",
        "L1": (
            "compared with a stated reference; how often it fails a correct fit is not "
            "measured, so no pass or fail"
        ),
        "L2": "pass or fail from an exact reference or a simulation study of this kind of fit",
        "L3": "as L2, with an external package's result beside it",
    }
)
