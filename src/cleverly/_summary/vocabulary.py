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
#: The study-setup and simulation summaries (X89 T5).
DESIGN: Final = "Design"
TREATMENT: Final = "Treatment"
BASELINE: Final = "Baseline"
CENSORING: Final = "Censoring"
FOLLOWED: Final = "Observed to the end"
EFFECTIVE_N: Final = "Effective sample size"
CENSORED_BEFORE_END: Final = "Censored before the end"
STRATA: Final = "Strata"
HELD_FIXED: Final = "Held fixed"
PROTOCOL: Final = "Study protocol"
ANSWERED_BY: Final = "Answered by"
METHODS_FIT: Final = "Methods that fit"
POPULATION: Final = "Population"
TIME_ZERO: Final = "Time zero"
INTERFERENCE_UNIT: Final = "Interference unit"
REPLICATIONS: Final = "Replications"
ROWS_PER_REPLICATION: Final = "Rows per replication"
INTERVAL_LEVEL: Final = "Interval level"
TOTAL_TIME: Final = "Total time"
ACCOUNTED: Final = "Accounted for"

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
        DESIGN,
        TREATMENT,
        BASELINE,
        CENSORING,
        FOLLOWED,
        EFFECTIVE_N,
        CENSORED_BEFORE_END,
        STRATA,
        HELD_FIXED,
        PROTOCOL,
        ANSWERED_BY,
        METHODS_FIT,
        POPULATION,
        TIME_ZERO,
        INTERFERENCE_UNIT,
        REPLICATIONS,
        ROWS_PER_REPLICATION,
        INTERVAL_LEVEL,
        TOTAL_TIME,
        ACCOUNTED,
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
        "sdr": "sequentially doubly robust regression (SDR)",
        "ipsi": "incremental propensity score intervention (IPSI)",
        "gcomp": "sequential g-computation",
        "ipw": "sequential inverse probability weighting",
        "parametric_gformula": "parametric g-formula",
        "riesz_tmle": "TMLE with a learned Riesz representer",
        "ep": "efficient plug-in learner",
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
        # The study-setup and simulation summaries (X89 T5).
        "further_conditions": "note",
        "checkable": "note",
        "model_restriction": "note",
        "several_readings": "note",
        "unbound_effect": "warning",
        "no_method": "warning",
        "other_methods": "note",
        "failed_replications": "warning",
        "measures_diagnostic": "warning",
        "mixed_statuses": "note",
        "worker_time": "note",
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
        # The study-setup and simulation summaries (X89 T5).
        "other_methods": (
            "The methods table lists the methods that fit. {count} other {methods} "
            "cannot estimate this question; the full summary says why."
        ),
        "further_conditions": (
            "The full summary states {count} further {conditions} the answer rests on."
        ),
        "checkable": (
            "The data cannot confirm these conditions. Overlap (marked partly) can be probed: "
            "after a fit, result.diagnostics.support() shows where the fitted treatment model "
            "gives units little chance of a compared treatment. That check reads a fitted "
            "model, so a clean result does not prove overlap."
        ),
        "model_restriction": (
            "The design restricts the model of the data; the full summary states how. The "
            "question is unchanged, and the interval of a fit is claimed only under the "
            "restriction."
        ),
        "several_readings": (
            "The answer has {count} causal readings, each resting on its own condition; the "
            "full summary states each."
        ),
        "unbound_effect": (
            "This question is not bound to data, so it cannot be estimated, and the methods "
            "table is worked out from the question without the study's design."
        ),
        "no_method": "No method fits this question; the full summary says why each does not.",
        "data_warning": "When cleverly read the data it warned: {messages}",
        "failed_replications": (
            "Replications that raised an error: {count} of {total}. Every column leaves "
            "them out, and the full summary lists the errors."
        ),
        "measures_diagnostic": (
            "One or more estimates supply no interval{some}, so mean SE, SE ratio, coverage "
            "and rejected describe the {noun} and not a confidence interval."
        ),
        "mixed_statuses": (
            "The replicates took more than one inference status, named as in the inference "
            "column of to_frame(): {clauses}."
        ),
        "worker_time": (
            "The parallel workers' times are each worker's own elapsed time, added across "
            "workers, so they can pass the time the fit waited and are not added to the "
            "table above."
        ),
        "spread_scale_simulation": (
            "For {names}, mean SE and SE ratio are on the log scale; truth, bias and MC SD "
            "are on the ratio scale."
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

# ----------------------------------------------------------------- study setup (T5)

#: The Design fact, keyed by the design class's name. ``{T}`` is the number of time points.
DESIGN_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PointTreatment": "one treatment decision",
        "LongitudinalTreatment": "treatment at {T} times",
        "TimeToEvent": "time to an event, {T} time points",
        "LongPanel": "{T} times, long rows read as one row per unit",
    }
)

#: The Answered by fact, keyed by whether the design is longitudinal.
ANSWERED_BY_WORDS: Final[Mapping[bool, str]] = MappingProxyType(
    {
        False: "adjusting for covariates",
        True: "adjusting for history",
    }
)
#: The Outcome fact of a time-to-event design.
EVENT_OUTCOME: Final = "{event} (event code), timed by {time}"
#: The Answered by fact of a randomized design that adjusts for nothing.
ANSWERED_BY_RANDOMIZED: Final = "randomized assignment"
#: The Adjusted for fact of a randomized design that adjusts for nothing.
RANDOMIZED_NO_ADJUSTMENT: Final = "none (randomized, as declared)"
#: The title of each setup summary.
STUDY_TITLE: Final = "Study design"
QUESTION_TITLE: Final = "Causal question ({name})"
METHODS_TITLE: Final = "Estimation methods"
PROTOCOL_TITLE: Final = "Study protocol"
SIMULATION_TITLE: Final = "Simulation study: {label}"
PHASES_TITLE: Final = "Time by phase"
#: Table titles.
ROLES_TITLE: Final = "Columns by role"
CONDITIONS_TITLE: Final = "Conditions the answer rests on"
ARMS_TITLE: Final = "Share of rows by treatment level"
#: The share column of that table, marked weighted on weighted data.
SHARE_COLUMN: Final = "share"
HISTORY_TITLE: Final = "What each treatment time adjusts for"
METHODS_TABLE_TITLE: Final = "Methods"
STRATEGIES_TITLE: Final = "Treatment strategies"
ELIGIBILITY_TITLE: Final = "Eligibility"
WORKERS_TITLE: Final = "In parallel workers"
#: Full-detail sections of the setup summaries.
SECTION_UNAVAILABLE: Final = "Why a method does not fit"
SECTION_INTERCURRENT: Final = "Intercurrent events"
SECTION_RATIONALE: Final = "Rationale for the assumptions"
SECTION_PER_ESTIMAND: Final = "Per estimand"
SECTION_FAILURES: Final = "Failed replications"
SECTION_COLUMNS: Final = "Reading the columns"
#: The lead of a question's Assumptions section, with the terms of art the short table's
#: plain names stand for.
CONDITIONS_LEAD: Final = (
    "The answer equals {functional} when the statements below hold. The short table calls "
    "consistency well-defined treatment, exchangeability no hidden confounding, and "
    "positivity overlap."
)

#: One condition's plain words: the name the short table prints, its sentence, and
#: whether the data can say anything about it.
Checkable = Literal["no", "partly"]

#: The plain words for the identification statements that recur, matched on the start of
#: the statement in this order, most specific first. Each value is ``(name, sentence,
#: checkable)``; ``{A}``, ``{Y}`` and ``{Z}`` are the treatment, outcome and held variable.
#: A statement no entry matches prints only in the full summary, verbatim, such as the
#: definedness of a ratio, a known policy or a model restriction. ``test_setup_summaries``
#: pins which entry each statement of a broad set of questions reaches.
CONDITION_WORDS: Final[tuple[tuple[str, tuple[str, str, Checkable]], ...]] = (
    (
        "consistency: each observed history",
        (
            "well-defined treatment",
            "Each unit's observed outcomes are the ones it would have had under the "
            "treatments it actually received over time. This needs each treatment to be "
            "well defined: every way of getting the same treatment has the same effect.",
            "no",
        ),
    ),
    (
        "consistency:",
        (
            "well-defined treatment",
            "Each unit's observed outcome is the outcome it would have had under the "
            "treatment it actually received{held}. This needs the treatment to be well "
            "defined: every way of getting the same level has the same effect.",
            "no",
        ),
    ),
    (
        "no interference:",
        ("no interference", "One unit's treatment does not change another unit's outcome.", "no"),
    ),
    (
        "no unmeasured intermediate confounding:",
        (
            "no hidden confounding of {Z}",
            "Among units with the same treatment and adjusted-for values, the value of {Z} "
            "is unrelated to the outcomes they would have had.",
            "no",
        ),
    ),
    (
        "no unmeasured treatment confounding:",
        (
            "no hidden confounding",
            "Among units with the same adjusted-for values, which treatment a unit got is "
            "unrelated to the outcomes it would have had. This holds when the adjusted-for "
            "columns account for the common causes of {A} and {Y} and none of them is "
            "affected by {A}.",
            "no",
        ),
    ),
    (
        "no unmeasured confounding:",
        (
            "no hidden confounding",
            "Among units with the same adjusted-for values, which treatment a unit got is "
            "unrelated to the outcomes it would have had. This holds when the adjusted-for "
            "columns account for the common causes of {A} and {Y} and none of them is "
            "affected by {A}.",
            "no",
        ),
    ),
    (
        "sequential exchangeability given the recorded history",
        (
            "no hidden confounding at each time",
            "At each time, among units with the same recorded history, which treatment a "
            "unit got and whether it stays in follow-up are unrelated to the outcomes it "
            "would have had.",
            "no",
        ),
    ),
    (
        "sequential exchangeability in the form the chosen interpretation names",
        (
            "no hidden confounding at each time",
            "At each time, among units with the same recorded history, the dose a unit got "
            "is unrelated to the outcomes it would have had, in the strong or the standard "
            "form that each causal reading names.",
            "no",
        ),
    ),
    (
        "dropout is missing at random given the recorded history",
        (
            "follow-up loss at random",
            "At each time, among units with the same recorded history, whether a unit stays "
            "in follow-up is unrelated to its later outcomes, and each unit still at risk had "
            "some chance of staying.",
            "no",
        ),
    ),
    (
        "sequential positivity for treatment and remaining under observation",
        (
            "overlap at each time",
            "At each time, units with every history that following the compared regimens "
            "can produce had some chance of the treatment a regimen assigns and of staying "
            "in follow-up.",
            "partly",
        ),
    ),
    (
        "positivity: the dose a policy assigns",
        (
            "overlap at each time",
            "At each time, units with every recorded history had some chance of the dose "
            "the policy assigns wherever they had a chance of their observed dose, and of "
            "staying in follow-up.",
            "partly",
        ),
    ),
    (
        "positivity: ",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of "
            "each compared level of {A}.",
            "partly",
        ),
    ),
    (
        "positivity for the evaluated arm:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "evaluated level of {A}.",
            "partly",
        ),
    ),
    (
        "positivity for the reference arm within each comparison population:",
        (
            "overlap",
            "Among units that got a compared level of {A}, every group that shares the "
            "adjusted-for values had some chance of the reference level.",
            "partly",
        ),
    ),
    (
        "positivity for each comparison arm within the reference population:",
        (
            "overlap",
            "Among units that got the reference level of {A}, every group that shares the "
            "adjusted-for values had some chance of each compared level.",
            "partly",
        ),
    ),
    (
        "positivity for the reference intervention:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "reference level of {A}.",
            "partly",
        ),
    ),
    (
        "positivity for the evaluated doses:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of each "
            "dose of {A} the question evaluates.",
            "partly",
        ),
    ),
    (
        "positivity for the evaluated treatment values:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of each "
            "treatment value the question evaluates.",
            "partly",
        ),
    ),
    (
        "positivity *for the regime*",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "level of {A} the rule assigns them.",
            "partly",
        ),
    ),
    (
        "positivity *for the assigned dose*",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "dose the policy assigns, wherever it had some chance of its observed dose.",
            "partly",
        ),
    ),
    (
        "positivity for the treated arm wherever the tilt keeps it",
        (
            "overlap",
            "Every group of units the tilt moves towards a level of {A} had some chance of "
            "that level.",
            "partly",
        ),
    ),
    (
        "intermediate positivity at",
        (
            "overlap for {Z}",
            "Every group of units that share the treatment and adjusted-for values had some "
            "chance of the value {Z} is held at.",
            "partly",
        ),
    ),
    (
        "missingness at random for",
        (
            "outcome missing at random",
            "Among units with the same treatment and adjusted-for values, whether {Y} was "
            "recorded is unrelated to its value.",
            "no",
        ),
    ),
    (
        "response positivity for",
        (
            "overlap for recorded outcomes",
            "Every group of units that share the treatment and adjusted-for values that the "
            "question reads had some chance of a recorded {Y}.",
            "partly",
        ),
    ),
    (
        "product positivity at the reference arm",
        (
            "overlap for recorded outcomes",
            "Every group of units that share the adjusted-for values had some chance of "
            "both the reference level of {A} and a recorded {Y}.",
            "partly",
        ),
    ),
)

#: The plain words that replace an entry of :data:`CONDITION_WORDS` when the design has a
#: feature: ``"held"`` a held intermediate, ``"randomized"`` a randomized assignment with no
#: adjustment, ``"uncensored"`` a longitudinal design without censoring.
CONDITION_VARIANTS: Final[Mapping[str, tuple[tuple[str, tuple[str, str, Checkable]], ...]]] = (
    MappingProxyType(
        {
            "held": (
                (
                    "missingness at random for",
                    (
                        "outcome missing at random",
                        "Among units with the same treatment, {Z} and adjusted-for values, "
                        "whether {Y} was recorded is unrelated to its value.",
                        "no",
                    ),
                ),
            ),
            "randomized": (
                (
                    "no unmeasured confounding:",
                    (
                        "no hidden confounding",
                        "Which treatment a unit got is unrelated to the outcomes it would "
                        "have had, as in a randomized trial. The design declares this, and "
                        "the data cannot show it.",
                        "no",
                    ),
                ),
                (
                    "positivity: ",
                    (
                        "overlap",
                        "Every unit had some chance of each compared level of {A}, as a "
                        "randomized assignment gives.",
                        "partly",
                    ),
                ),
            ),
            "uncensored": (
                (
                    "sequential exchangeability given the recorded history",
                    (
                        "no hidden confounding at each time",
                        "At each time, among units with the same recorded history, which "
                        "treatment a unit got is unrelated to the outcomes it would have had.",
                        "no",
                    ),
                ),
                (
                    "sequential positivity for treatment and remaining under observation",
                    (
                        "overlap at each time",
                        "At each time, units with every history that following the compared "
                        "regimens can produce had some chance of the treatment a regimen "
                        "assigns.",
                        "partly",
                    ),
                ),
                (
                    "positivity: the dose a policy assigns",
                    (
                        "overlap at each time",
                        "At each time, units with every recorded history had some chance of "
                        "the dose the policy assigns wherever they had a chance of their "
                        "observed dose.",
                        "partly",
                    ),
                ),
            ),
        }
    )
)

#: The words a question's held intermediate adds to the well-defined-treatment sentence.
HELD_CLAUSE: Final = " and the value of {Z} it was held at"


#: The reading of a simulation study.
SIMULATION_PASS: Final = (
    "Coverage is not below {level} by more than Monte Carlo error, and the bias is within "
    "Monte Carlo error of zero."
)
SIMULATION_DIAGNOSTIC_PASS: Final = (
    "The bias is within Monte Carlo error of zero. The coverage column measures a {noun}, so "
    "it certifies no confidence interval."
)
SIMULATION_UNDERCOVERAGE: Final = (
    "{name}: coverage {coverage} is below the nominal {level} by more than Monte Carlo error"
)
SIMULATION_SMALL_UNDERCOVERAGE: Final = (
    "{name}: coverage {coverage} is below the nominal {level} by more than Monte Carlo error, "
    "though by no more than {floor} percentage points"
)
#: One estimand's clause of the mixed-statuses note.
MIXED_STATUS_CLAUSE: Final = "{name} took {statuses}, and each of its columns reads {what}"
SIMULATION_SE_SMALL: Final = "; the {spread} is {factor}x too small"
SIMULATION_SE_ZERO: Final = "; every reported standard error is zero"
SIMULATION_SE_RIGHT: Final = (
    "; the standard error is not too small, so the shortfall points to bias or a skewed "
    "sampling distribution"
)
SIMULATION_BIAS: Final = "{name}: bias {bias} is {count} Monte Carlo standard errors from zero"
SIMULATION_CONSTANT_BIAS: Final = (
    "{name}: every replicate returned the same estimate, so its bias {bias} has no Monte Carlo "
    "error"
)
SIMULATION_SMALL_BIAS: Final = (
    "{name}: bias {bias} is {count} Monte Carlo standard errors from zero, though no more "
    "than {floor} of the larger of 1 and the absolute truth"
)
SIMULATION_EXPERIMENTAL: Final = (
    "At least one replicate's estimate of {names} has no validation study behind its kind of "
    "fit, so cleverly marks it experimental."
)
#: How to read the columns of a phase timing profile.
PHASE_COLUMNS: Final[tuple[str, ...]] = (
    "own s: the seconds spent in the phase itself, outside any phase nested in it; this "
    "column adds up to the accounted time.",
    "with nested s: the seconds from entering the phase to leaving it, nested phases "
    "included, so this column overlaps itself and does not add up.",
    "share: own seconds as a share of the total time.",
)
