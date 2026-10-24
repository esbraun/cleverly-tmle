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
HEADER_LABELS: frozenset[str] = frozenset(
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
NOTE_CODES: Mapping[str, NoteKind] = MappingProxyType(
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
        "model_reliance": "note",
        "structural_model": "note",
        "check_splits": "note",
        "draw_order": "note",
        "simulation_share": "note",
        "unsupported_rows": "warning",
        "score_load": "note",
        "score_load_rows": "note",
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
RESTATED_WARNINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
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

NOTE_TEXTS: Mapping[str, str] = MappingProxyType(
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
            "this can make the interval too narrow; cross-fitting (n_folds of two or more, or "
            "cross_fitting=CrossFitting() on a study) gives out-of-fold predictions."
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
            "tolerance{draw}, so the estimate and its {what} may be off. "
            "result.diagnostics.score_equations() gives each time point."
        ),
        "score_rule_unmet_longitudinal": (
            "The fit did not meet its declared stopping rule{draw}: {misses}. The estimate and "
            "its {what} may be off by more than the rule allows; "
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
        "model_reliance": "{statement}",
        "structural_model": (
            "The intervals hold under the linear structural nested mean model with the blip "
            "features declared at each time point. A blip that depends on an undeclared "
            "earlier treatment or covariate makes the coefficients, and the plan means read "
            "from them, biased."
        ),
        "structural_covariate_feature": (
            "A blip feature reads a covariate measured after baseline, so the fit reports the "
            "blips and the reference mean but no plan means or contrasts."
        ),
        "structural_check": "result.check_model() compares the plan contrasts with LTMLE.",
        "check_experimental": "cleverly marks this check experimental. {reason}",
        "check_splits": (
            "Each number is the mean over {k} fold splits, the first of them the fit's own."
        ),
        "draw_order": (
            "The order in which the covariates of one time point are drawn defines the "
            "models, so at most one order is correctly specified in general."
        ),
        "simulation_share": (
            "The simulation contributes up to {share} of the variance ({name}); a larger "
            "n_draws lowers it."
        ),
        "unsupported_rows": (
            "Rows given an arm whose estimated probability is exactly zero: {counts}. The data "
            "give no support there. If the true probability is zero, the effect is not "
            "identified for those rows."
        ),
        "score_load": (
            "Score load reads the limited treatment model the fit used, while the ratios and "
            "their effective size read the estimated model before any limit. It counts the "
            "rows that effectively carry the estimating equation, of the rows the equation "
            "reads, so it is not the effective size of the ratios."
        ),
        "score_load_rows": (
            "Score load counts the rows that effectively carry the estimating equation, of the "
            "rows the equation reads, so it is not the effective size of the ratios."
        ),
        "unsupported_density": (
            "{name} moves {count} rows to a dose where the estimated density is exactly zero. "
            "This flags the density model; it does not prove that the effect is unidentified."
        ),
    }
)

# ----------------------------------------------------------------- full detail

SECTION_ASSUMPTIONS: Final = "Assumptions"
SECTION_SETTINGS: Final = "Settings"
SECTION_METHOD: Final = "Method details"
SECTION_STUDY: Final = "Study description"
SECTION_HOW_TO_READ: Final = "How to read"
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

#: The structural nested mean summary (X89 T2): the Question, the row meanings and the
#: blip-feature table.
STRUCTURAL_QUESTION: Final = "effect of each treatment time on {Y} under the blip model"
STRUCTURAL_QUESTION_PLANS: Final = (
    "effect of each treatment time on {Y} under the blip model, and the plan means it implies"
)
STRUCTURAL_LEVEL: Final = "{A} = {level} instead of {ref}"
STRUCTURAL_DOSE: Final = "one more unit of {A}"
STRUCTURAL_MEANINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "blip": (
            "coefficient of {feature} in the effect on {Y} of {change} at time {t}, with later "
            "treatments at their reference"
        ),
        "intercept": (
            "effect on {Y} of {change} at time {t}, with later treatments at their reference, "
            "for a unit whose other blip features are zero"
        ),
        "reference": "mean {Y} if every unit had the reference level of every treatment",
        "level": "mean {Y} if every unit followed plan {d}, read from the blip model",
        "contrast": (
            "mean {Y} under plan {d} minus that mean under plan {r}, read from the blip model"
        ),
    }
)
BLIP_TITLE: Final = "Blip features at each time point"
TIME: Final = "time"
BLIP_TREATMENT: Final = "treatment"
BLIP_LEVELS: Final = "levels"
BLIP_FEATURES: Final = "features"
STRUCTURAL_FAMILY: Final = "Outcome regressions: {family}."
FAMILY_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"binomial": "logistic, for a 0-or-1 outcome", "gaussian": "linear, for a continuous outcome"}
)
STRUCTURAL_EIGENVALUE: Final = (
    "Smallest standardized eigenvalue of the estimating equation's derivative: {value}. Near "
    "zero, the treatment varies little given the history, or the blip features are nearly "
    "collinear given it, so the coefficients are weakly identified."
)
STRUCTURAL_EXTREME: Final = (
    "Largest share of rows with an estimated treatment probability within {margin} of 0 or 1, "
    "over the time points: {share}."
)
STRUCTURAL_RESTRICTION: Final = (
    "the effect of each treatment time, with later treatments at their reference, is linear "
    "in the declared blip features"
)
STRUCTURAL_CONSTRUCTION: Final = (
    "G-estimation: at each time point, from the last to the first, the blip coefficients solve "
    "an estimating equation that pairs the outcome, with later blips removed, against the "
    "treatment minus its predicted value."
)

#: The overlap reports of regimes, policies and tilts.
REGIME_ROW_TITLE: Final = "Overlap of regime {name}"
SUPPORT_REGIMES: Final = "Regimes"
SUPPORT_NO_REGIMES: Final = "The fit declares no regimes, so there is no overlap to report."
SUPPORT_WORST: Final = "Least supported"
SUPPORT_REGIME_TABLE: Final = "How much the data support each regime"
SUPPORT_REGIME_ONE_TABLE: Final = "How much the data support the regime"
SUPPORT_POLICY_TABLE: Final = "How much the data support the policy"
SUPPORT_TILT_TABLE: Final = "How much the data support the tilt"
SUPPORT_SPLIT: Final = " (score load from split {split} of {repeats})"
SUPPORT_REGIME: Final = "regime"
SUPPORT_MIN_G: Final = "min probability"
SUPPORT_MAX_RATIO: Final = "max ratio"
SUPPORT_RATIO_ESS: Final = "effective n"
SUPPORT_LOAD_EFFECTIVE: Final = "score load"
SUPPORT_UNSUPPORTED: Final = "unsupported"
SUPPORT_TWO_MECHANISMS: Final = (
    "min probability is the smallest estimated probability of an arm the regime assigns, max "
    "ratio the largest weight ratio, and effective n the effective sample size of those ratios. "
    "All three read the estimated treatment model before any limit, which is what the data "
    "support. unsupported counts the rows given an arm whose estimated probability is exactly zero."
)
SUPPORT_REGIME_DETAIL: Final = (
    "{name}: weight ratio quantiles {quantiles}; probability of the assigned arm {tails}."
)
SUPPORT_MOVED: Final = "Rows moved"
SUPPORT_UNSUPPORTED_ROWS: Final = "Rows sent to zero density"
SUPPORT_NOT_MEASURED: Final = "not measured"
SUPPORT_OUTSIDE: Final = "Rows sent outside the declared support"
SUPPORT_MECHANISM_NOTE: Final = (
    "The weight is the density ratio divided by the probability of staying under observation "
    "after its bound, and by the intermediate's density when the fit holds one; the smallest "
    "other factor is their smallest product."
)
SUPPORT_CAPPED: Final = "Rows held by the cap"
SUPPORT_MIN_DENSITY: Final = "Smallest density at an observed dose"
SUPPORT_NO_DENSITY: Final = "not measured: the classifier fits no density"
SUPPORT_MIN_MECHANISM: Final = "Smallest other factor dividing the weight"
SUPPORT_RATIO: Final = "density ratio"
SUPPORT_WEIGHT: Final = "weight"
SUPPORT_QUANTILE: Final = "{q} quantile"
SUPPORT_LARGEST: Final = "largest"
SUPPORT_ESS_ROW: Final = "effective size"
SUPPORT_SHARE_OF_ROWS: Final = "effective share"
SUPPORT_MEAN_RATIO: Final = "mean ratio"
SUPPORT_FOLD_MEAN: Final = "mean ratio, fold {fold}"
SUPPORT_MEAN_RATIO_READING: Final = (
    "The mean ratio's expected value under the true density equals the probability that the "
    "policy's dose stays inside the support of the observed doses given the covariates, which is "
    "one when the policy keeps that support; an upper cap alone does not ensure this for a "
    "negative shift or a gap in the support. In a sample the mean also moves with sampling and "
    "with error in the density model, so a mean near one does not prove that the support holds "
    "or that the effect is identified."
)
SUPPORT_ATOMS: Final = "Mean fitted probability of each declared atom: {masses}."
SUPPORT_NO_ZERO_CHECK: Final = (
    "The classifier fits no density, so the report cannot count rows sent to a dose of zero "
    "density."
)
SUPPORT_FACTORS: Final = "Odds factors"
SUPPORT_BOUND: Final = "Tilt weight bound"
SUPPORT_NO_FACTOR: Final = "none (the natural course)"
SUPPORT_TILT_READING: Final = (
    "An odds tilt keeps the tilt part of each weight inside the bound for any treatment "
    "model, so it needs no treatment probability away from 0 or 1. When outcomes are missing, "
    "the probability of staying under observation still divides the weight and needs "
    "positivity; the largest weight and the effective size do not include it. min "
    "probability is shown for comparison with a fit that sets each unit's treatment."
)

#: The parametric g-formula simulation report.
YES: Final = "yes"
NO: Final = "no"
GFORMULA_DRAWS: Final = "Draws per unit (estimate)"
GFORMULA_DERIVATIVE_DRAWS: Final = "Draws per unit (derivatives)"
GFORMULA_MODELS: Final = "Models"
GFORMULA_PARAMETRIC: Final = "All models parametric"
GFORMULA_MODELS_TITLE: Final = "The models the simulation draws from"
GFORMULA_MODEL: Final = "model"
GFORMULA_ROLE: Final = "role"
GFORMULA_RESPONSE: Final = "response"
GFORMULA_LEARNER: Final = "learner"
GFORMULA_READ: Final = "in estimate"
GFORMULA_CONSTANT: Final = "constant"
#: Short learner names for the models table, which has six columns.
GFORMULA_LEARNERS: Final[Mapping[str, str]] = MappingProxyType(
    {"GLM": "GLM", "LogisticRegression": "logistic", "LinearRegression": "linear"}
)
GFORMULA_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "covariate": "covariate",
        "outcome": "outcome",
        "hazard": "event hazard",
        "treatment": "natural treatment",
    }
)
GFORMULA_RESPONSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "binary": "0 or 1",
        "multi": "categories",
        "normal": "normal",
        "mean": "mean",
    }
)
GFORMULA_SHARE_TITLE: Final = "Share of each row's variance from the simulation"
GFORMULA_SHARE: Final = "share"
GFORMULA_SUPPORT_TITLE: Final = "Support of each set dose"
GFORMULA_COLUMN: Final = "treatment"
GFORMULA_DOSE: Final = "dose"
GFORMULA_RULE_DOSE: Final = "set by a rule"
GFORMULA_OFF: Final = "histories > {z} SD off"
GFORMULA_WINDOW: Final = "observed within {z} SD"
GFORMULA_READ_MODELS: Final = (
    "The simulation draws each unit's covariates and outcome forward in time from these "
    "models, under each plan. in estimate says whether the estimate draws from the model: a "
    "set dose replaces the treatment model, which is then fitted but not used."
)
GFORMULA_READ_SHARE: Final = (
    "share is the part of each row's variance that comes from simulating a finite number of "
    "draws per unit rather than from the data; a larger n_draws lowers it."
)
GFORMULA_READ_SUPPORT: Final = (
    "For each set dose, the first share counts simulated histories whose fitted mean dose lies "
    "more than {z} fitted standard deviations from the dose, and the second the observed doses "
    "within {z} standard deviations of it. A large first share or a small second one says the "
    "dose lies outside what the data show, so the estimate rests on the models' extrapolation."
)

#: The structural model check summary.
CHECK_QUESTION: Final = "do the model's plan contrasts agree with LTMLE on the same rows?"
CHECK_CONTRASTS: Final = "Contrasts"
CHECK_SPLITS_LABEL: Final = "Fold splits"
CHECK_TEST: Final = "Test"
CHECK_TEST_F: Final = "scaled joint Wald statistic against F({df}, {df2})"
CHECK_TEST_CHI2: Final = "joint Wald statistic against chi-squared({df})"
CHECK_TABLE_TITLE: Final = "The model's contrasts against LTMLE"
CHECK_STRUCTURAL: Final = "structural"
CHECK_LTMLE: Final = "LTMLE"
CHECK_DIFFERENCE: Final = "difference"
CHECK_SPREAD: Final = "std. error"
CHECK_JOINT_TITLE: Final = "Joint test"
CHECK_STATISTIC: Final = "statistic"
CHECK_CONTRASTS_COLUMN: Final = "contrasts"
CHECK_COVARIANCE_DF: Final = "covariance df"
CHECK_PVALUE: Final = "p-value"
CHECK_READING: Final = (
    "A small p-value says the model fails for these contrasts; a large one does not confirm "
    "the model."
)
CHECK_NO_TEST: Final = "No test: {reason}"
CHECK_HOW_TO_READ: Final[tuple[str, ...]] = (
    "The check fits LTMLE on the same rows, weights, clusters and folds and compares each plan "
    "contrast with the one the structural model implies. When the model holds, each "
    "difference has mean zero. The two estimates share most of their error, so the test reads "
    "the standard error of the difference, not that of either estimate.",
    "covariance df is the effective degrees of freedom of the estimated covariance. The test "
    "scales the Wald statistic and reads it against an F reference whose second degrees of "
    "freedom come from it, which allows for a covariance estimated with error. A small value "
    "says a few rows or clusters carry most of the variance, or few fold splits were averaged.",
)

#: The small summaries of an MSM fit's working-model record and an SNMM's positivity record.
MSM_FIT_TITLE: Final = "Working-model coefficients"
MSM_TABLE: Final = "Coefficient of each term"
MSM_TERM: Final = "term"
MSM_COEFFICIENT: Final = "coefficient"
MSM_LINK: Final = "Link"
MSM_LINK_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"identity": "the mean", "log": "the log of the mean", "logit": "the log odds"}
)
MSM_CELLS: Final = "Plan and time cells"
MSM_CONVERGED: Final = "Converged"
MSM_CAUSE_LABEL: Final = "Cause"
MSM_FIT_READING: Final = "The result's summary() gives each coefficient with its interval."
MSM_HOW_TO_READ: Final = (
    "The coefficients project the plan means at every plan and time onto the working model, "
    "which is linear in {scale}. They summarize those means, so they keep their meaning when "
    "the working model is wrong."
)
DIAGNOSTICS_TITLE: Final = "Positivity of the structural fit"
DIAGNOSTICS_TABLE: Final = "What each time point measured"
DIAGNOSTICS_NODE: Final = "treatment time"
DIAGNOSTICS_TIME: Final = "time {time}"
DIAGNOSTICS_EIGENVALUE: Final = "smallest eigenvalue"
DIAGNOSTICS_EXTREME: Final = "near 0 or 1"
DIAGNOSTICS_CONSTANT: Final = "Fits with a one-class fold"
DIAGNOSTICS_READING: Final = (
    "smallest eigenvalue is the smallest standardized eigenvalue of the estimating "
    "equation's derivative at that time: near zero, the treatment varies little given the "
    "history, or the blip features are nearly collinear given it, so the coefficients are "
    "weakly identified. near 0 or 1 is the share of rows whose estimated treatment "
    "probability lies within the positivity margin of 0 or 1; a "
    "dose has none."
)

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

#: A competing-risks working-model coefficient names its cause.
MSM_CAUSE: Final = "{text}, for {cause}"

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
IN_ONE_SPLIT: Final = ", in at least one split"

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
PLAN_SHIFT: Final = "the observed treatment {delta}{cap}"
PLAN_SCALE: Final = "the observed treatment times {factor}{cap}"
PLAN_NAMED_MTP: Final = "policy {name}"
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

# ----------------------------------------------------------------- point-estimate objects (T1)

#: Header labels of the summaries of a single estimate, a test, a bootstrap, a cross-fitted
#: report and the method records a point fit keeps.
ESTIMAND: Final = "Estimand"
NULL_VALUE: Final = "Null value"
SCALE: Final = "Scale"
REFERENCE: Final = "Reference"
RESAMPLING: Final = "Resampling"
REPLICATES: Final = "Replicates asked for"
ESTIMANDS_COVERED: Final = "Estimands"
FAILED: Final = "Failed"
INTERVAL: Final = "Interval"
FOLDS: Final = "Folds"
SPLITS: Final = "Splits"
CANDIDATES: Final = "Candidates"
CHOSEN: Final = "Chosen"
CRITERION: Final = "Criterion"
SEARCH: Final = "Search"
COVARIATES_SEARCHED: Final = "Covariates searched"
ADJUSTMENT: Final = "Multiple testing"
LOSS: Final = "Loss"
BEST: Final = "Best candidate"
ENSEMBLE_RISK: Final = "Ensemble CV risk"
GUARD: Final = "Extra equations for"
REDUCTION: Final = "Extra regressions"
FORM: Final = "Form"
ROUNDS: Final = "Rounds"
STOPPED: Final = "Stopped"
DESIGN: Final = "Design"
STRATA: Final = "Strata"
SOURCE: Final = "Source"
TRAINING_SETS: Final = "Training sets"
INTERMEDIATE: Final = "Intermediate"
CRITICAL_VALUE: Final = "Critical value"
MULTIPLIER_DRAWS: Final = "Multiplier draws"
MULTIPLIER_KIND: Final = "Multiplier weights"
STATISTIC: Final = "statistic"

HEADER_LABELS = HEADER_LABELS | {
    ESTIMAND,
    NULL_VALUE,
    SCALE,
    REFERENCE,
    RESAMPLING,
    REPLICATES,
    ESTIMANDS_COVERED,
    FAILED,
    INTERVAL,
    FOLDS,
    SPLITS,
    CANDIDATES,
    CHOSEN,
    CRITERION,
    SEARCH,
    COVARIATES_SEARCHED,
    ADJUSTMENT,
    LOSS,
    BEST,
    ENSEMBLE_RISK,
    GUARD,
    REDUCTION,
    FORM,
    ROUNDS,
    STOPPED,
    DESIGN,
    STRATA,
    SOURCE,
    TRAINING_SETS,
    INTERMEDIATE,
    CRITICAL_VALUE,
    MULTIPLIER_DRAWS,
    MULTIPLIER_KIND,
}

#: Titles of the T1 summaries.
ESTIMATE_ONE_TITLE: Final = "Estimate: {name}"
WALD_TITLE: Final = "Wald test: {name}"
BOOTSTRAP_TITLE: Final = "Bootstrap replicates"
BOOTSTRAP_INTERVAL_TITLE: Final = "Bootstrap interval"
BANDS_TITLE: Final = "Simultaneous {level} bands"
SPREAD_TITLE: Final = "Spread across cross-fitting splits"
SEARCH_TITLE: Final = "Collaborative treatment-model search"
ENSEMBLE_TITLE: Final = "Super learner"
IMPORTANCE_TITLE: Final = "Variable importance"
OAT_TITLE: Final = "Outcome-adaptive treatment model"
DERIVATIVE_TITLE: Final = "Collaborative TMLE, derivative form"
REDUCED_TITLE: Final = "DR-TMLE extra regressions"
IPTW_TITLE: Final = "Adaptive IPTW fit"
CV_TITLE: Final = "Cross-validated estimates"
RESULT_SET_TITLE: Final = "Controlled direct effects ({method})"
#: The Question of a result set over several levels of the intermediate variable.
RESULT_SET_QUESTION: Final = "effect of {A} on {Y} with {Z} held at each level below for every unit"

#: The parts of an outcome-adaptive per-arm treatment model, keyed by the start of the
#: factor label ``CTMLEOutcomeAdaptiveFit.factors`` records.
OAT_FACTORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "P(A=a": "treatment",
        "P(Delta=1": "outcome recorded",
        "P(Delta_A=1": "treatment recorded",
    }
)
#: Where a derivative-form fit took its two treatment models, keyed by ``source``.
DERIVATIVE_SOURCES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lasso": "lasso path",
        "learners": "two declared learners",
        "arrays": "two declared probability arrays",
    }
)
#: The two forms of an Adaptive IPTW estimate, keyed by ``form``.
IPTW_FORMS: Final[Mapping[str, str]] = MappingProxyType(
    {"targeted": "targeted", "one_step": "one-step"}
)
#: Why the Adaptive IPTW weight adjustment stopped, keyed by ``stop``.
IPTW_STOPS: Final[Mapping[str, str]] = MappingProxyType(
    {"tolerance": "converged", "cap": "round limit"}
)
#: The spread columns of a cross-validated report, keyed by its frame column.
CV_SPREAD_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "cv_std_err": "CV std. error",
        "cv_plugin_std_err": "CV SE (diagnostic)",
        "pooled_std_err": "std. error",
        "pooled_plugin_std_err": "plug-in SE (diagnostic)",
    }
)
#: The row label of a one-row bootstrap table.
REPLICATES_ROW: Final = "replicates"
#: The kind of a bootstrap interval, keyed by ``BootstrapSummary.interval``.
BOOTSTRAP_INTERVAL_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"percentile": "percentile", "bca": "BCa"}
)
#: What a bootstrap resamples, keyed by ``BootstrapResult.resampling``.
RESAMPLING_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"iid": "rows", "cluster": "clusters"}
)

#: The row label of a result set's estimate at one level of the intermediate variable.
LEVEL_ROW: Final = "{name} at {Z} = {z}"

#: Plain words for the scale an estimate's interval is built on, keyed by ``scale``.
SCALE_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "difference": "difference",
        "level": "level",
        "fraction": "fraction",
        "ratio": "ratio (interval on the log scale)",
    }
)

#: Plain words for each collaborative search, keyed by the ``strategy=`` of ``CTMLE``.
CTMLE_SEARCHES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "greedy": "adding covariates one at a time",
        "ordered": "adding covariates in a fixed order",
        "discrete": "over declared covariate sets",
        "lasso": "along a lasso penalty path",
        "learners": "over declared learners",
        "oat": "outcome-adaptive",
        "derivative": "derivative form",
    }
)

#: The order an ``ordered`` search adds covariates in, keyed by ``preorder``.
CTMLE_PREORDERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "logistic": "ranked by a logistic fit of the treatment",
        "partial_correlation": "ranked by partial correlation with the outcome residual",
        "custom": "as declared",
    }
)

#: The losses a collaborative search and a super learner score candidates by.
LOSS_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "loglik": "negative log-likelihood",
        "squared": "squared error",
        "mse": "squared error",
        "squared_error": "squared error",
        "nll": "negative log-likelihood",
        "neg_log_likelihood": "negative log-likelihood",
        "log_loss": "negative log-likelihood",
        "negative_log_likelihood": "negative log-likelihood",
    }
)

#: What each DR-TMLE guard protects against, keyed by the guard name.
GUARD_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Q": "a wrong outcome model",
        "g": "a wrong treatment model",
    }
)

#: What the extra equations of each DR-TMLE guard are shown to do, keyed by the guard
#: name. ``drtmle.py``'s module docstring states the conditions: the one-wrong interval is
#: valid for a known mechanism, conservative for a correctly specified parametric one, not
#: established for a data-adaptive learner, and not established at all with the mechanism
#: wrong.
GUARD_SENTENCES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Q": (
            "The fit solved extra equations meant to keep the interval valid when the "
            "outcome model is wrong. With the outcome model wrong, the interval is valid when "
            "the treatment probabilities are known, conservative (wider than needed) when "
            "they come from a correctly specified parametric model, and not established for "
            "a flexible learner."
        ),
        "g": (
            "The fit solved the extra equations meant for a wrong treatment model. When the "
            "treatment model is wrong, no result shows that the interval keeps its coverage."
        ),
    }
)

#: The Method details line of a DR-TMLE fit whose design restricts the treatment model to
#: ``{names}``: when the restriction fails, the treatment model is wrong, and the
#: mechanism-wrong rows of the one-wrong table of the DR-TMLE theorem page decide the interval.
DRTMLE_RESTRICTED: Final = (
    "The treatment model reads only {names}, as the design declares. If treatment depends "
    "on other covariates too, the treatment model is wrong, and no result shows that the "
    "interval keeps its coverage."
)
#: The Method details lines of an Adaptive IPTW fit whose design restricts the treatment
#: model. It fits no outcome model, so it is consistent only when the omitted covariates
#: confound nothing beyond ``{names}``.
IPTW_RESTRICTED: Final = "The treatment model reads only {names}, as the design declares."
IPTW_RESTRICTED_OMITTED: Final = (
    " The estimate is consistent only when {omitted} add no confounding beyond those."
)
#: The Method details line of a collaborative search drawn from a declared pool.
SEARCH_POOL: Final = "The search drew only from {names}, as the design declares."

#: The extra regressions of each DR-TMLE construction, keyed by ``reduction``.
REDUCTION_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "univariate": "three one-dimensional",
        "bivariate": "two, one of them two-dimensional",
        "missing_outcome": "five, for missing outcomes",
    }
)

#: Plain names of the DR-TMLE extra regressions, keyed by ``ReducedFit.diagnostics`` key.
REDUCED_REGRESSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "qr": "outcome error on treatment probability",
        "gr1": "treatment on outcome prediction",
        "gr1_bivariate": "treatment on outcome and treatment predictions",
        "gr2": "weighted treatment error on outcome prediction",
        "gamma_a": "treatment on outcome prediction",
        "gamma_m": "outcome recorded on outcome prediction",
        "r_a": "weighted treatment error on outcome prediction",
        "r_m": "weighted recording error on outcome prediction",
        "e": "outcome error on P(treated and recorded)",
    }
)

#: The column labels of the T1 tables.
REPLICATE_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "usable": "usable",
        "mean": "mean",
        "sd": "SD of replicates",
        "smallest": "smallest",
        "largest": "largest",
    }
)
SPREAD_DIAGNOSTIC: Final = "spread (diagnostic)"
SPREAD_RATIO: Final = "spread / SE"
SPREAD_RATIO_PLUGIN: Final = "spread / plug-in SE"
SPREAD_PLUGIN_SE: Final = "plug-in SE"
FOLD_AVERAGED: Final = "fold-averaged"
WHOLE_SAMPLE: Final = "whole-sample"
FOLD_SD: Final = "fold SD"
FOLD_RANGE: Final = "fold range"
ADJUSTED_P_VALUE: Final = "adjusted p-value"
CV_RISK: Final = "CV risk"
IN_SAMPLE_RISK: Final = "in-sample risk"
WEIGHT: Final = "weight"
TREATMENT_MODEL_COLUMN: Final = "treatment model"
LOWER: Final = "lower"
UPPER: Final = "upper"
#: The marks a search table puts beside the candidate the fit reports.
CHOSEN_MARK: Final = "chosen"
FINAL_MARK: Final = "final step"

#: More note codes, used by the point-estimate objects (thread T1).
T1_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "rounds_capped": "warning",
        "levels_separate": "note",
        "null_scale": "note",
        "bootstrap_failed": "warning",
        "bootstrap_describes": "note",
        "split_spread": "note",
        "spread_missing": "note",
        "cv_two_reports": "note",
        "cv_first_split": "note",
        "left_out": "note",
        "search_stopped": "note",
        "risk_order": "warning",
        "stacked_interval": "note",
        "ordinary_tmle": "note",
        "multiple_testing": "note",
        "separate_fits": "note",
        "learners_failed": "warning",
        "ensemble_risk": "note",
        "fit_cautions": "warning",
        "stacked_refused": "note",
        "rows_shown": "note",
        "guard_scope": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T1_NOTE_CODES})

T1_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "rounds_capped": (
            "The weight adjustment stopped at its round limit ({rounds}) before the "
            "weights converged for every estimate, so an estimate may keep part "
            "of the bias the adjustment is meant to remove."
        ),
        "levels_separate": (
            "Each level of {Z} is a different parameter. The levels do not split the total "
            "effect into direct and indirect parts, and their difference is an interaction "
            "contrast, not a mediated effect."
        ),
        "null_scale": (
            "The test compares the estimate with the null value on the {scale} scale, where "
            "the interval is built."
        ),
        "bootstrap_failed": (
            "{failed} of {requested} replicates gave no usable estimate and are left out. If "
            "those were the resamples where a treatment group was rarest, the remaining "
            "spread, and any interval from it, can be too narrow."
        ),
        "bootstrap_describes": (
            "These numbers describe the replicate estimates. The fitted result's summary "
            "says whether they give a confidence interval for this fit."
        ),
        "bootstrap_describes_one": (
            "These numbers describe the replicate estimates of one row. The fitted result's "
            "summary says whether the interval is a confidence interval for this fit."
        ),
        "split_spread": (
            "Each spread is the standard deviation of the estimate across {repeats} random "
            "splits into folds. It shows how much the split moved the estimate; it is a "
            "diagnostic, not a standard error. {tail}"
        ),
        "spread_missing": (
            "{names} {verb} no spread: a split gave a value that is not finite on the scale "
            "of the interval."
        ),
        "cv_two_reports": (
            "The fold-averaged estimate evaluates the parameter inside each fold and "
            "averages the folds, with {spread}. The whole-sample estimate puts the held-out "
            "predictions together and evaluates them once. They agree for the means and the "
            "average effect when every fold carries the same total weight, and they can "
            "differ for ratios and for effects among the treated or untreated."
        ),
        "cv_first_split": (
            "Both estimates combine {repeats} splits by their median; the fold columns "
            "describe the first split only."
        ),
        "left_out": (
            "The chosen treatment model leaves out {names}. The search chose by the "
            "cross-validated criterion of the adjusted outcome predictions; it does not show "
            "why a covariate was left out."
        ),
        "search_stopped": (
            "The search stopped after candidate {stopped}: {patience} since the best one so "
            "far had a higher cross-validated risk, which is the stopping rule. Later "
            "candidates were {later}."
        ),
        "search_never_stopped": (
            "The stopping rule (patience {patience}) never triggered, so every candidate "
            "was scored."
        ),
        "risk_order": (
            "The declared learners are not ordered from least to most adaptive: the "
            "treatment-model risk rises at candidate {names}. The search assumes that order."
        ),
        "ordinary_tmle": (
            "No extra regressions were fitted (an empty guard), so the estimate is the "
            "ordinary TMLE."
        ),
        "multiple_testing": (
            "The adjusted p-values control the false discovery rate over the {count} tests "
            "({method}) if the tests are independent or positively dependent, which these "
            "fits do not guarantee. The intervals and the unadjusted p-values are not "
            "adjusted."
        ),
        "separate_fits": (
            "Each row comes from its own fit with that candidate as the treatment; the "
            "full summary lists what each fit adjusts for. Each row needs its own no-unmeasured-"
            "confounding and positivity conditions, and a row is not the candidate's share "
            "of one effect."
        ),
        "learners_failed": "These candidates failed to fit and were left out: {names}.",
        "ensemble_risk": (
            "Each candidate's CV risk is scored on held-out folds. The ensemble's weights "
            "were chosen on those same held-out predictions, so its CV risk is optimistic "
            "beside the candidates' risks."
        ),
        "fit_cautions": (
            "The fits of {names} carry warnings that change how to read their rows; print "
            "each with result.fits[name].summary()."
        ),
        "stacked_interval": (
            "Each interval is pointwise and comes from one stacked estimating equation of "
            "the declared parametric outcome and treatment models{weights}. It covers the "
            "arm mean when the outcome model is correct; otherwise it covers the value the "
            "estimate settles at, which can move with the sample size while a limit on the "
            "treatment probabilities binds."
        ),
        "stacked_weights": (
            ", with the data weights in every equation, so the interval treats the weights as fixed"
        ),
        "stacked_weights_estimated": (
            ", with the data weights in every equation, so the interval treats the weights "
            "as fixed although they are declared estimated"
        ),
        "stacked_refused": "No stacked interval: {reason}",
        "rows_shown": (
            "The table shows {shown} of the {total} candidates: the first, the last, the "
            "chosen one and each one whose covariates change. selection.to_frame() lists "
            "them all."
        ),
        "spread_withheld": (
            "No interval for {names}, so the std. error column is empty there; the spread "
            "is still a diagnostic of the split."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T1_NOTE_TEXTS})

#: The end of the ``split_spread`` note, by whether the fit supplies an interval.
SPLIT_SPREAD_TAILS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "inference": "The reported std. error already includes it.",
        "withheld": ("The fit gives no interval, so the plug-in SE beside it is a diagnostic too."),
    }
)
#: What the fold-averaged estimate of a cross-validated report carries, by the same rule.
CV_SPREAD_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "inference": "the cross-validated standard error",
        "withheld": (
            "a cross-validated spread that is a diagnostic here, as the fit gives no interval"
        ),
    }
)
#: The Adaptive IPTW cap warning ``rounds_capped`` restates.
RESTATED_WARNINGS = MappingProxyType(
    {
        **RESTATED_WARNINGS,
        "rounds_capped": ("before the relative score reached tol=",),
        "risk_order": ("strategy='learners' needs treatment_candidates= ordered",),
    }
)
