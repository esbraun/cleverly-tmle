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
#: The unit count of a fit read from a long panel, which has one row per unit and time.
UNITS: Final = "Units"
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
#: The censoring indicators of a longitudinal fit, and what every plan does with them.
CENSORING: Final = "Censoring"
ANALYSIS: Final = "Analysis"
#: The longitudinal and structural summaries (X89 T2).
TREATMENT: Final = "Treatment"
COVARIATE_MODELS: Final = "Covariate models"
CONTRASTS: Final = "Contrasts"
FOLD_SPLITS: Final = "Fold splits"
TEST: Final = "Test"
DRAWS: Final = "Draws per unit"
DERIVATIVE_DRAWS: Final = "Draws per unit (derivatives)"
MODELS: Final = "Models"
ALL_PARAMETRIC: Final = "All models parametric"
REGIMES: Final = "Regimes"
LEAST_SUPPORTED: Final = "Least supported"
POLICY: Final = "Policy"
MOVED: Final = "Rows moved"
CAPPED: Final = "Rows held by the cap"
SMALLEST_DENSITY: Final = "Smallest density"
FACTORS: Final = "Factors"
WEIGHT_BOUND: Final = "Weight bound"
SMALLEST_PROBABILITY: Final = "Smallest probability"

#: Every header label above, so a test can check a summary uses only shared labels.
HEADER_LABELS: Final[frozenset[str]] = frozenset(
    {
        QUESTION,
        UNITS,
        CENSORING,
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
        TREATMENT,
        COVARIATE_MODELS,
        CONTRASTS,
        FOLD_SPLITS,
        TEST,
        DRAWS,
        DERIVATIVE_DRAWS,
        MODELS,
        ALL_PARAMETRIC,
        REGIMES,
        LEAST_SUPPORTED,
        POLICY,
        MOVED,
        CAPPED,
        SMALLEST_DENSITY,
        FACTORS,
        WEIGHT_BOUND,
        SMALLEST_PROBABILITY,
    }
)

# ----------------------------------------------------------------- column labels

ESTIMATE: Final = "estimate"
P_VALUE: Final = "p-value"
DF: Final = "df"
SUPPORT: Final = "support"
EXPERIMENTAL: Final = "experimental"
#: The rows an experimental note names when every row of the table is experimental.
EVERY_ROW: Final = "every row"
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
        # The longitudinal, structural and support summaries (X89 T2).
        "bootstrap_bands": "note",
        "model_based_interval": "note",
        "working_model": "note",
        "isotonic_moves": "note",
        "structural_model": "note",
        "check_splits": "note",
        "draw_order": "note",
        "simulation_share": "note",
        "model_reliance": "note",
        "two_mechanisms": "note",
        "unsupported_rows": "warning",
        "outside_declared_support": "warning",
        "mean_ratio": "note",
        "tilt_bound": "note",
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
            "Cumulative mechanism truncation exceeded",
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
        # The longitudinal, structural and support summaries (X89 T2).
        "in_sample_flexible_longitudinal": (
            "The models were fitted and evaluated on the same rows. With flexible learners "
            "this can make the interval too narrow; n_folds of two or more gives out-of-fold "
            "predictions."
        ),
        "weights_capped_cumulative": (
            "cleverly limited the estimated probability of following plan {plan} through "
            "time {time} to [{low}, {high}] for {share} of the units still following it "
            "then{split}."
        ),
        "weights_capped_paths": (
            "The probability limits [{low}, {high}] moved {share} of the followed paths of "
            "plan {plan} that start at time {time}{split}."
        ),
        "weights_capped_cumulative_effect": (
            "A limited probability caps the weight of the unit it moves. That steadies the "
            "estimate, and it can move the estimate if the true probabilities lie outside the "
            "limits. result.diagnostics.truncation_curve(bounds) refits under other limits."
        ),
        "score_unsolved_longitudinal": (
            "At some time point the fit did not solve its estimating equation within "
            "tolerance{draw}, so the estimate and its interval may be off. "
            "result.diagnostics.score_equations() gives each time point."
        ),
        "score_rule_unmet_longitudinal": (
            "The fit did not meet its declared stopping rule{draw}: {misses}. The estimate and "
            "its interval may be off by more than the rule allows; "
            "result.diagnostics.score_equations() gives each time point."
        ),
        "bootstrap_bands": (
            "The bootstrap bands cover all {count} rows at once with {level} probability. They "
            "come from {replicates} resamples."
        ),
        "model_based_interval_gcomp": (
            "The interval holds only if every outcome model is correctly specified. Under a "
            "wrong model it covers the limit of that model, not the plan mean. LTMLE and SDR "
            "stay consistent when either all outcome models or all treatment and censoring "
            "models are right."
        ),
        "model_based_interval_ipw": (
            "The interval holds only if every treatment and censoring model is correctly "
            "specified. Under a wrong model it covers the limit of that model, not the plan "
            "mean. LTMLE and SDR stay consistent when either all outcome models or all "
            "treatment and censoring models are right."
        ),
        "model_based_interval_gformula": (
            "The interval holds only if every covariate model, outcome (or event-hazard) model "
            "and natural-treatment model the simulation reads is correctly specified. It "
            "includes the simulation error."
        ),
        "known_factors": (
            "The factors declared known ({factors}) use the declared probabilities and need no "
            "model."
        ),
        "working_model": (
            "Each coefficient comes from projecting the plan means onto the working model "
            "{terms} ({link} link). The coefficients are defined whether or not that model "
            "holds, and they are its causal coefficients only when it does."
        ),
        "isotonic_moves": (
            "cleverly made each curve non-decreasing, which moved {count} levels ({names}). "
            "The standard errors are unchanged; an interval at a moved level is valid only "
            "where the true curve strictly increases."
        ),
        "structural_model": (
            "The intervals hold under the linear structural nested mean model with the blip "
            "features declared for each time point. A blip that depends on an undeclared "
            "earlier treatment or covariate makes the coefficients answer a different question. "
            "result.check_model() compares the plan contrasts with LTMLE."
        ),
        "check_splits": (
            "Each number is the mean over {splits} fold splits, the first of them the fit's own."
        ),
        "draw_order": (
            "The order in which the covariates of one time point are drawn defines the "
            "models, so at most one order is correctly specified in general."
        ),
        "simulation_share": (
            "The simulation contributes up to {share} of the variance ({name}); a larger "
            "n_draws lowers it."
        ),
        "model_reliance": "{statement}",
        "two_mechanisms": (
            "The smallest probability, the largest ratio and its effective sample size read the "
            "estimated treatment model before any limit, which is what the data support. The "
            "score load reads the limited model the fit used, counted as the rows that "
            "effectively carry the estimating equation."
        ),
        "unsupported_rows": (
            "{name} gives {count} rows an arm whose estimated probability is exactly zero, so "
            "the effect is not identified for those rows under the fitted model."
        ),
        "unsupported_density": (
            "{name} moves {count} rows to a dose where the estimated density is exactly zero. "
            "This flags the density model; it does not prove that the effect is unidentified."
        ),
        "outside_declared_support": (
            "{name} assigns {count} rows a dose outside the declared support."
        ),
        "mean_ratio": (
            "Under the true density the mean ratio equals the probability that the policy's "
            "dose stays inside the support of the observed doses given the covariates, which "
            "is one when the policy keeps that support. In a sample it also moves with sampling "
            "and with error in the density model."
        ),
        "tilt_bound": (
            "An odds tilt keeps every weight inside the bound, whatever the treatment model, so "
            "it needs no treatment probability away from zero or one. The smallest probability is "
            "shown for comparison with a fit that sets every unit's treatment."
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

# ----------------------------------------------------------------- longitudinal (T2)

#: The plain name of each longitudinal estimator, keyed by ``LongitudinalConfig.estimator``.
LONGITUDINAL_METHOD_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "longitudinal targeted maximum likelihood (LTMLE)",
        "msm": "longitudinal TMLE of a working model (MSM)",
        "sdr": "sequentially doubly robust estimator (SDR)",
        "ipsi": "incremental propensity score estimator (IPSI)",
        "gcomp": "sequential g-computation",
        "ipw": "inverse probability weighting (IPW)",
        "parametric_gformula": "parametric g-formula (forward simulation)",
    }
)

#: The short name a longitudinal summary title carries.
LONGITUDINAL_SHORT_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "LTMLE",
        "msm": "LTMLE, working model",
        "sdr": "SDR",
        "ipsi": "IPSI",
        "gcomp": "g-computation",
        "ipw": "IPW",
        "parametric_gformula": "parametric g-formula",
    }
)

#: The IPW normalization in words, keyed by ``LongitudinalConfig.normalization``.
IPW_FORMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "hajek": "Hajek form, normalized by the weight sum",
        "horvitz_thompson": "Horvitz-Thompson form, normalized by n",
    }
)

#: The title of a longitudinal summary, filled with the short method name.
LONGITUDINAL_TITLE: Final = "Longitudinal estimate ({method})"
STRUCTURAL_TITLE: Final = "Structural blip estimates (SNMM)"
STRUCTURAL_METHOD: Final = "linear structural nested mean model (SNMM)"
CHECK_TITLE: Final = "Structural model check"
SIMULATION_TITLE: Final = "Simulation report (parametric g-formula)"
REGIME_SUPPORT_TITLE: Final = "Overlap of each regime"
POLICY_SUPPORT_TITLE: Final = "Overlap of policy {name}"
TILT_SUPPORT_TITLE: Final = "Overlap of tilt {name}"

#: What one row of a longitudinal table estimates, keyed by the name's prefix and the
#: outcome kind. ``{Y}`` is the outcome (with ``{at}`` its time, empty at the end of study),
#: ``{d}`` the plan, ``{r}`` the reference plan, ``{t}`` the time and ``{cause}`` the cause.
LONGITUDINAL_MEANINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ey": "mean {Y}{at} if every unit followed plan {d}",
        "ate": "mean {Y}{at} if every unit followed plan {d}, minus that mean under plan {r}",
        "risk": "probability of the event by time {t} if every unit followed plan {d}",
        "risk_ate": (
            "probability of the event by time {t} if every unit followed plan {d}, minus "
            "that probability under plan {r}"
        ),
        "cif": "probability of {cause} by time {t} if every unit followed plan {d}",
        "cif_ate": (
            "probability of {cause} by time {t} if every unit followed plan {d}, minus "
            "that probability under plan {r}"
        ),
        "msm": "coefficient of {term} in the working model of the plan means",
    }
)

#: The Censoring header fact: the indicators, and that every plan keeps each unit observed,
#: so a row is its value had no unit been censored.
CENSORING_FACT: Final = "{names}; each plan keeps every unit observed"
#: The Outcome fact of a competing-risks fit.
COMPETING_FACT: Final = "{causes} (competing causes)"

#: The Question fact of a single-row longitudinal table, keyed as the meanings are.
LONGITUDINAL_QUESTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ey": "mean {Y}{at} under plan {d}",
        "ate": "average effect on {Y}{at} of plan {d} instead of {r}",
        "risk": "risk of the event by time {t} under plan {d}",
        "risk_ate": "change in the risk of the event by time {t}, plan {d} instead of {r}",
        "cif": "risk of {cause} by time {t} under plan {d}",
        "cif_ate": "change in the risk of {cause} by time {t}, plan {d} instead of {r}",
        "msm": "working-model coefficient of {term}",
    }
)

#: The Question fact when a longitudinal table reports several rows.
LONGITUDINAL_SEVERAL_ROWS: Final = "{Y} under each plan; each row is defined below"
#: The outcome a several-row Question names on a fit with an outcome at every time point.
REPEATED_OUTCOME: Final = "the outcome at each time point"
#: The Question fact of a working-model fit.
LONGITUDINAL_MSM_QUESTION: Final = (
    "working-model projection of the plan means of {Y}; each row is defined below"
)

#: The start of the fit-time warning a longitudinal ``weights_capped`` note restates.
CUMULATIVE_TRUNCATION_WARNING: Final = "Cumulative mechanism truncation exceeded"
#: Where a longitudinal note's largest share came from, on a fit with repeated splits.
IN_ONE_SPLIT: Final = ", in one split"

#: The titles and column labels of the longitudinal tables.
PLANS_TITLE: Final = "What each plan assigns and who followed it to the end"
PLANS_SPLIT_TITLE: Final = (
    "What each plan assigns and who followed it to the end (split one of {repeats})"
)
PLAN: Final = "plan"
ASSIGNS: Final = "assigns"
IN_PLAN: Final = "followed"
#: The count column of a tilt plan, which keeps every unit at risk: the units observed.
OBSERVED_TO_END: Final = "observed"
FOLLOWED_IN_DATA: Final = "followed in data"
LARGEST_WEIGHT: Final = "max weight"
EFFECTIVE_N: Final = "effective size"
BOOTSTRAP_BANDS_TITLE: Final = "Bootstrap sup-t {level} bands"

#: Plain words for one node of a plan.
PLAN_RULE: Final = "a declared rule"
PLAN_NAMED_RULE: Final = "rule {name}"
PLAN_POLICY: Final = "a known random draw"
PLAN_NAMED_POLICY: Final = "random draw {name}"
PLAN_TILT: Final = "odds of treatment times {delta}"
PLAN_TILT_LEVELS: Final = "odds of each level times {factors}"
PLAN_NATURAL: Final = "the data's own treatment process"
PLAN_EVERY_TIME: Final = "{words} at every time point"
PLAN_AT_TIME: Final = "{words} at time {t}"
PLAN_HELD: Final = "{words} from baseline, held"

#: The conditions a longitudinal fit with no study rests on, in plain words.
LONGITUDINAL_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "consistency: each observed history equals the potential history under the plan it followed",
    "no interference: one unit's outcome does not depend on other units' plans",
    "sequential exchangeability: at every time point, treatment and remaining under "
    "observation are independent of the potential outcomes given the recorded history",
    "sequential positivity: at every time point, every unit on a plan has a positive "
    "probability of the plan's treatment and of remaining under observation",
)
#: What an odds tilt needs in place of treatment positivity (Kennedy 2019; Kim, Kennedy and
#: Naimi 2021).
TILT_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "an odds tilt needs no treatment positivity at a tilt time point, because it keeps the "
    "support of each arm (Kennedy 2019, Theorem 1); a fixed, rule or random-draw time point "
    "still needs positivity wherever its plan puts mass",
    "dropout is missing at random given the recorded history, and each unit at risk remains "
    "under observation with positive probability (Kim, Kennedy and Naimi 2021, Assumptions "
    "A2-M and A3)",
)
#: The construction of each estimator, with its source, for the Method details section.
LONGITUDINAL_CONSTRUCTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": (
            "A backward sequence of outcome regressions, each updated so that its estimating "
            "equation holds, from the last time point to the first."
        ),
        "sdr": (
            "Sequentially doubly robust estimation: a backward sequence of regressions of a "
            "transformed outcome, with no separate update step (Diaz, Williams, Hoffman and "
            "Schenck 2023, Section 5.3)."
        ),
        "ipsi": (
            "The mean of the influence function at the fitted models (Kennedy 2019, Algorithm "
            "1; Kim, Kennedy and Naimi 2021, Algorithm 1, under censoring). An odds tilt moves "
            "each unit's treatment probability pi to delta pi / (delta pi + 1 - pi)."
        ),
        "gcomp": (
            "The iterated conditional expectation plug-in (Bang and Robins 2005, Section 4), "
            "with no update step."
        ),
        "ipw": "Each unit's outcome weighted by the inverse probability of staying on the plan.",
        "parametric_gformula": (
            "Forward simulation of the fitted covariate, outcome and hazard models under each "
            "plan (Robins 1986; McGrath et al. 2020), with no update step."
        ),
    }
)
#: The caveat on a rule's follower counts, for the Settings section.
POLICY_FOLLOWERS: Final = (
    "A plan that draws each unit's treatment at random counts as followed by the units whose "
    "observed treatment the draw can give."
)
RULE_FOLLOWERS: Final = (
    "A rule's followers depend on their covariates, so the counts in the plan table describe "
    "this sample."
)
#: The ratio-diagnostic column labels of the support tables.
SUPPORT_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "regime": "regime",
        "min_propensity": "smallest probability",
        "max_ratio": "largest ratio",
        "effective_n": "effective sample size",
        "score_load": "score load",
        "unsupported": "rows without support",
    }
)
