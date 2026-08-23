"""Plain-English names for everything a study's committed results refer to.

The result files identify a test by key: a scenario, an estimand, a property family, a cell.  Those
keys are precise and unreadable.  ``tests/studies/evidence/document.py`` renders one documentation
row per committed test, and this module supplies the column that says what the test actually
checked.

The descriptions are written by hand.  Everything beside them in a rendered row is read from the
artefacts, so a description is the one part of a published table a person chooses.
``tests/unit/test_method_evidence.py::TestThePublishedTestTables`` requires every key present in a
committed result file to resolve here.  A study cannot publish a row whose meaning is undeclared.

Deliberately not listed in any :class:`~tests.studies.evidence.registry.StudyRecord` ``modules``
tuple.  That tuple is hashed into each ``manifest.json``, and nothing here can change a measured
result -- only how it reads.
"""

from __future__ import annotations

import re

#: ``estimand`` values carry a regimen in brackets for the longitudinal studies.
_PARAMETERISED = re.compile(r"^(?P<name>[a-z_]+)\[(?P<argument>.+)\]$")

#: Longitudinal property cells prefix the plan they belong to.
_ARM = re.compile(r"^(?P<arm>static|dynamic)__(?P<cell>.+)$")


IMPLEMENTATIONS: dict[str, str] = {
    "cleverly": "`cleverly`",
    "cleverly-ctmle-oat": "`cleverly` outcome-adaptive C-TMLE",
    "cleverly-ctmle-selector": "`cleverly` selector-based C-TMLE",
    "cleverly-fold-evaluated-cvtmle": "`cleverly` fold-evaluated CV-TMLE",
    "cleverly-stacked-cvtmle": "`cleverly` stacked CV-TMLE",
    "ltmle": "R `ltmle`",
    "r-ctmle": "R `ctmle`",
    "tlverse-ctmle3-oat": "R `ctmle3`",
    "tmle3": "R `tmle3`",
    "tmle3-cvtmle": "R `tmle3` CV-TMLE",
}


SCENARIOS: dict[str, str] = {
    "binary": "binary-outcome law",
    "binary_discrete": "binary-outcome law, discrete selector",
    "binary_greedy": "binary-outcome law, greedy selector",
    "binary_ordered": "binary-outcome law, ordered selector",
    "censored_end_of_study": "two-time-point law with monotone censoring",
    "continuous": "bounded continuous-outcome law with effect modification",
}


ESTIMANDS: dict[str, str] = {
    "atc": "average effect on the untreated",
    "ate": "average treatment effect",
    "att": "average effect on the treated",
    "ey0": "counterfactual mean under no treatment",
    "ey1": "counterfactual mean under treatment",
    "ey_obs": "observed outcome mean under the natural course",
    "or": "marginal odds ratio, reported on the log scale",
    "paf": "population attributable fraction",
    "par": "population attributable risk",
    "rr": "marginal risk ratio, reported on the log scale",
}

#: The bracketed half of a longitudinal estimand key.
REGIMENS: dict[str, str] = {
    "always": "treat at both times",
    "never": "treat at neither time",
    "treat then continue if l2 positive": "treat, then continue only if L2 is positive",
}

#: What the parameterised estimand names mean once the regimen is resolved.
PARAMETERISED: dict[str, str] = {
    "ey_regimen": "mean outcome under the plan",
    "ate_regimen": "difference in mean outcome between the plans",
}


PROPERTIES: dict[str, str] = {
    "crossfit_overfitting": (
        "cross-fitting removes the optimism a flexible learner puts into an in-sample fit"
    ),
    "double_robustness": (
        "the estimator stays consistent when either the outcome regression or the treatment "
        "mechanism is correct"
    ),
    "generated_design": (
        "the outcome-adaptive design costs precision when its regression is estimated rather "
        "than known"
    ),
    "interval_calibration": (
        "the reported standard error matches the sampling spread and the interval covers near "
        "its nominal rate"
    ),
    "power": "the test detects a real effect, so a passing null result cannot come from an inert test",
    "robustness_contract": (
        "the estimator stays consistent under the one nuisance the method's source paper claims"
    ),
    "root_n_and_efficiency": (
        "bias, coverage and standard-error calibration hold across sample sizes"
    ),
    "root_n_rate": "the sampling spread contracts at the root-n rate the theory predicts",
    "selector_necessity": "the collaborative selector is what produces the result, not the fit around it",
    "targeting_necessity": "the targeting step is what produces the result, not the plug-in beneath it",
    "type_i_error": "under a confounded sharp null the test rejects no more often than its nominal size",
}


#: ``(family, cell)`` after any arm prefix is stripped, to ``(what was tested, what must hold)``.
CELLS: dict[tuple[str, str], tuple[str, str]] = {
    ("crossfit_overfitting", "cross_fitted_oat"): (
        "outcome-adaptive C-TMLE with cross-fitted nuisances and a flexible learner",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "fold_evaluated_cvtmle"): (
        "fold-evaluated CV-TMLE with a flexible learner",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "stacked_cvtmle"): (
        "stacked CV-TMLE with a flexible learner",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "in_sample_control"): (
        "the same flexible learner fitted in sample, with no cross-fitting",
        "SE ratio must fall below the overfitting ceiling",
    ),
    ("double_robustness", "both_correct"): (
        "both the outcome regression and the treatment mechanism are correctly specified",
        "bias interval inside the equivalence margin",
    ),
    ("double_robustness", "outcome_correct"): (
        "only the outcome regression is correctly specified",
        "bias interval inside the equivalence margin",
    ),
    ("double_robustness", "treatment_correct"): (
        "only the treatment mechanism is correctly specified",
        "bias interval inside the equivalence margin",
    ),
    ("double_robustness", "mechanism_correct"): (
        "only the treatment and censoring mechanisms are correctly specified",
        "bias interval inside the equivalence margin",
    ),
    ("double_robustness", "both_wrong"): (
        "both nuisances are misspecified",
        "bias interval must fall entirely outside the margin",
    ),
    ("generated_design", "oracle_design"): (
        "the outcome-adaptive design is supplied rather than estimated",
        "SE ratio interval inside the calibration band",
    ),
    ("generated_design", "estimated"): (
        "the same design is estimated from the data, as a real fit does",
        "the SE-ratio deficit must reach the declared shortfall",
    ),
    ("interval_calibration", "correctly_specified"): (
        "both nuisances are correctly specified",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "shrunken_se_control"): (
        "the reported standard errors are multiplied by a declared factor below one",
        "the SE-ratio interval must fall below the calibration band",
    ),
    ("interval_calibration", "noise_control"): (
        "one efficiency-bound unit of independent noise is added to each estimate",
        "the empirical efficiency ratio must rise above the band",
    ),
    ("power", "alternative"): (
        "the same test applied to a law with a real effect",
        "rejection lower bound clears the minimum power",
    ),
    ("robustness_contract", "outcome_correct"): (
        "the outcome regression is correct and the mechanism is not",
        "bias interval inside the equivalence margin",
    ),
    ("robustness_contract", "outcome_wrong"): (
        "the outcome regression is misspecified",
        "bias interval must fall entirely outside the margin",
    ),
    ("root_n_and_efficiency", "n_500"): (
        "bias, coverage and SE calibration at n = 500",
        "the requirements depend on whether this rung is a positive or small-sample control",
    ),
    ("root_n_and_efficiency", "n_2000"): (
        "bias, coverage and SE calibration at n = 2,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_and_efficiency", "n_8000"): (
        "bias, coverage and SE calibration at n = 8,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_rate", "empirical_sd"): (
        "log empirical spread of the estimates regressed on log n across three sizes",
        "slope interval inside the root-n band and excluding -1/4",
    ),
    ("root_n_rate", "reported_se"): (
        "the same regression applied to the mean reported standard error",
        "slope interval inside the root-n band and excluding -1/4",
    ),
    ("selector_necessity", "collaborative"): (
        "the selector chooses its own mechanism path",
        "bias interval inside the equivalence margin",
    ),
    ("selector_necessity", "empty_control"): (
        "the selector is forced to stop at an empty path",
        "bias interval must fall entirely outside the margin",
    ),
    ("targeting_necessity", "targeted"): (
        "the estimator fluctuates a constant outcome model, so targeting does all the adjusting",
        "bias interval inside the equivalence margin",
    ),
    ("targeting_necessity", "untargeted"): (
        "the identical backward recursion with no fluctuation at any node",
        "bias interval must fall entirely outside the margin",
    ),
    ("type_i_error", "sharp_null"): (
        "a confounded law whose true contrast is exactly zero",
        "one-sided rejection bound stays under the declared type-I ceiling",
    ),
}

#: Longitudinal cells belong to one of the two plans a study contrasts.
ARMS: dict[str, str] = {"static": "static plan", "dynamic": "dynamic plan"}


class Undescribed(LookupError):
    """A committed result names a key this module does not describe."""


def implementation(key: str) -> str:
    """The reader-facing name of an implementation column value."""
    try:
        return IMPLEMENTATIONS[key]
    except KeyError:
        raise Undescribed(f"no description for implementation {key!r}") from None


def scenario(key: str) -> str:
    """The law a scenario samples."""
    try:
        return SCENARIOS[key]
    except KeyError:
        raise Undescribed(f"no description for scenario {key!r}") from None


def estimand(key: str) -> str:
    """What an estimand key names, including a bracketed longitudinal regimen."""
    match = _PARAMETERISED.match(key)
    if match is None:
        try:
            return ESTIMANDS[key]
        except KeyError:
            raise Undescribed(f"no description for estimand {key!r}") from None
    name, argument = match.group("name"), match.group("argument")
    if name not in PARAMETERISED:
        raise Undescribed(f"no description for parameterised estimand {name!r}")
    return f"{PARAMETERISED[name]} {regimen(argument)}"


def regimen(argument: str) -> str:
    """A regimen label, or a contrast written as one label against another."""
    if " vs " in argument:
        treated, reference = argument.split(" vs ", 1)
        return f'"{regimen(treated)}" against "{regimen(reference)}"'
    try:
        return REGIMENS[argument]
    except KeyError:
        raise Undescribed(f"no description for regimen {argument!r}") from None


def claim(family: str) -> str:
    """The property a family tests."""
    try:
        return PROPERTIES[family]
    except KeyError:
        raise Undescribed(f"no description for property family {family!r}") from None


def cell(
    family: str, key: str, *, exact_efficiency: bool = False, role: str | None = None
) -> tuple[str, str]:
    """What a property cell configures, and what its verdict requires.

    The arm prefix a longitudinal study puts on a cell name selects the plan rather than the
    configuration, so it is stripped and reported beside the shared description.
    """
    arm = _ARM.match(key)
    base = arm.group("cell") if arm else key
    try:
        tested, required = CELLS[family, base]
    except KeyError:
        raise Undescribed(f"no description for cell {key!r} of {family!r}") from None
    if family == "interval_calibration" and base == "correctly_specified" and exact_efficiency:
        tested += " with an independently computed efficiency bound"
        required += ", with both efficiency-ratio intervals inside their bands"
    if family == "root_n_and_efficiency" and base == "n_500":
        if role == "positive":
            required = "bias inside the margin, coverage clears the floor, SE ratio inside the band"
        elif role == "control":
            required = "coverage interval lies below nominal or clears the declared floor"
    if arm is not None:
        tested = f"{ARMS[arm.group('arm')]}: {tested}"
    return tested, required
