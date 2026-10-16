r"""Registered evidence for the plug-in omitted-variable bound's standard error (X26).

The subject is the one-sided limits of the bias-adjusted bound (Chernozhukov, Cinelli, Newey,
Sharma and Syrgkanis 2026, Theorem 4) under ``nu2_estimator="plugin"``.  The package reports
the second-order corrected plug-in :math:`\hat\nu^2 = E_n[\hat\alpha^2] - \hat B`, with a
corrected variance, and refuses a fit on which the correction does not hold
(``truncation_active``, ``correction_unstable``, ``corrected_nonpositive``).  The curve of
:math:`\hat\nu^2` stacks the score of a declared parametric mechanism with the second moment
(Saul and Hudgens 2020, *J. Stat. Softw.* 92(2), Section 2, Equations (1) and (2)).  A declared
known mechanism has no mechanism term.

**What the study claims, and what it only measures.**  The claim cells are the lower and the
upper bound of each parameter.  Each reads the bound as an estimate of a known population value,
with the standard error read back off the reported limit as in the RM22 study.  The corrected
:math:`\nu^2` rows are not claims.  At ``n = 1000`` the corrected :math:`\nu^2` interval
under-covers (``tests/diagnostics/x26_crossfit_bias`` measures a standard-error ratio of 0.73 to
0.75 cross-fitted and 0.90 to 0.92 in sample), and no accessor of the package publishes it.  The
study records each draw's corrected :math:`\nu^2`, its corrected standard error, the raw
:math:`E_n[\hat\alpha^2]` and :math:`\hat B` in the primary rows' extra columns, and
:func:`extra_artifacts` publishes them as an unjudged reading (:data:`NU2_ROWS`,
:data:`NU2_READING`).  Nothing gates that reading.

**The scenarios.**  Every scenario fits ``n = 1000`` rows with a ``LinearRegression`` outcome
and ``estimands=("ate", "att", "atc")``.

``binary_logistic``
    :func:`cleverly.datasets.make_linear_ate`, an in-sample TMLE and
    ``LogisticRegression(C=numpy.inf, max_iter=2000)``.
``binary_logistic_crossfit``
    The same with ``cross_fit=True, n_folds=5``.  A cross-fitted continuous outcome needs a
    declared outcome support, so every cross-fitted scenario passes
    ``q_bounds=(-40.0, 45.0)``, the support of ``tests/diagnostics/x26_crossfit_bias``.  The
    binary outcome is ``2 + 1.5 A`` plus a normal term with standard deviation about 1.746, so
    each bound lies more than 23 standard deviations from both arm means.  No draw reaches it,
    and the bounds only fix the scale of the logistic fluctuation.
``binary_logistic_ridge`` and ``binary_logistic_ridge_crossfit``
    The same with ``LogisticRegression(max_iter=1000)``: a fixed L2 penalty with ``C = 1``,
    the idiom of the documentation.
``multinomial_logistic`` and ``multinomial_logistic_crossfit``
    The three-arm law :func:`draw_multinomial`.  :math:`W \sim N(0, I_3)`, the logits
    :math:`(0, 0.3 W_1 - 0.2 W_2, -0.2 W_1 + 0.3 W_2)`, and
    :math:`Y = 0.6\,\mathrm{step}[A] + W_1 - 0.5 W_2 + 0.2 W_3 + N(0, 1)` with steps
    ``(0, 1, 2.4)``.  ``LogisticRegression(C=numpy.inf, max_iter=2000)``.  The package's
    three-arm law :func:`cleverly.datasets.make_multi_arm` is not used: its :math:`E[1/g^3]` is
    about :math:`e^{14}`, so the curve of :math:`\hat\nu^2` has a standard deviation of about
    709 and no calibration cell could resolve it.
``binary_known``
    ``make_linear_ate`` with its own mechanism,
    ``cleverly.datasets.synthetic.linear_dgp().propensity``, declared through
    ``treatment_probabilities=``.  The mechanism term is zero.
``binary_unbalanced``
    The ``make_linear_ate`` outcome law and covariates with the treatment intercept moved to
    :math:`-1.5` (:func:`draw_unbalanced`), so about 19 percent of rows are treated.  In sample,
    ``LogisticRegression(C=numpy.inf, max_iter=2000)``.  The lower arm of the mechanism sits
    nearer the truncation bound, where the package's ``truncation_active`` check acts
    (``tests/unit/test_omitted_variable_plugin_design.py``, design check (b)).

**The truths.**  ``sigma^2 = 1`` on every law.  On ``make_linear_ate`` the closed forms of the
RM22 study hold: :math:`\nu^2 = 2 + 2e^{0.07}` for the ATE and :math:`4e^{0.07}` for the ATT
and the ATC, and the effect is 1.5.  The unbalanced law has the same effect and covariate
variance; its :math:`\nu^2` is a one-dimensional Gauss-Hermite rule over the logit-normal
index.  On the three-arm law a two-dimensional rule with 80 nodes per axis gives
:math:`p_a = E[g_a]`, :math:`E[1/g_a + 1/g_r]`, :math:`(p_a + E[g_a^2/g_r]) / p_a^2` and
:math:`(p_r + E[g_r^2/g_a]) / p_r^2`; the effects are 1.44, 1.44 and 0.6.

**The independent rebuild.**  Each replicate rebuilds the curve of :math:`\hat\nu^2` from the
captured coefficients, the folds and the data, and refuses the replicate unless it equals the
package's curve within :data:`REBUILD_TOLERANCE`.  It reads no package curve and imports no
package code of the stack.  The bound's reported standard error is the root of the curve's
variance less the package's variance correction, times the delta-method coefficient of the
bound, and the study rebuilds the first and the coefficient.  :math:`\hat B` and the variance
correction themselves are package values.  The exact-law witnesses of
``tests/unit/test_omitted_variable_plugin_second_order.py`` check them, and this study does not
rebuild them.  Each control is the package's curve minus (inflated) or plus (shrunken) a
mechanism term that this module computed.  It keeps the positive's *relative* variance
correction, :math:`(v_+ - c^2 v) / v_+`, so the control's variance stays positive whenever the
package's own is (:func:`_control_standard_errors`).  A control therefore never fails a replicate
that the package returned.

**The failure accounting.**  A replicate is one parameter of one draw.  A refusal of one
parameter raises :class:`~cleverly.exceptions.DataError`, which the driver records as a failed
replicate of that parameter's cells and of no other parameter's.  Only a refusal of the package
raises it.  A defect of the study stops the run (:func:`parameter_rows`).  The primary phase therefore
registers one scenario for each parameter (:func:`primary_name`), and the three parameters of a
scenario read one sample through ``scenario_seed_owners``.

The X26 plan (rev 3.1, section 3.5 and section 8; rev 4.2, section 9.10) declared this study,
its margins, its budget and its red-cell route before any run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_linear_ate
from cleverly.datasets.synthetic import linear_dgp
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import omitted_variable as omitted_variable_module
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 10_000
PRIMARY_N = 1_000
SEED = 20262601
RESAMPLING_SEED = 20262602

#: The strength of the X26 contract.  At (0.5, 0.3) the ATT control's predicted ratio, 1.089,
#: sits too close to the band edge 1.07 (plan section 3.4), so the study raises ``cf_d``.
CF_Y = 0.5
CF_D = 0.5
RHO = 1.0

#: The quantities each parameter reports.  ``lower`` and ``upper`` are the claim rows.
#: ``nu2`` is a measured, unjudged reading.
ROWS = ("nu2", "lower", "upper")
CLAIM_ROWS = ("lower", "upper")

#: The parameters of each law: the study's token and the package's name, and the
#: representer's group, arm and the arm it is contrasted with.
BINARY_PARAMETERS: dict[str, tuple[str, str, float, float]] = {
    "ate": ("ate", "mean", 1.0, 0.0),
    "att": ("att", "att", 1.0, 0.0),
    "atc": ("atc", "atc", 1.0, 0.0),
}
MULTINOMIAL_PARAMETERS: dict[str, tuple[str, str, float, float]] = {
    "ate_2v0": ("ate[2.0 vs 0.0]", "mean", 2.0, 0.0),
    "att_2v0": ("att[2.0 vs 0.0]", "att", 2.0, 0.0),
    "atc_1v0": ("atc[1.0 vs 0.0]", "atc", 1.0, 0.0),
}

#: Each scenario's law, its treatment mechanism (``logistic``, ``ridge`` or ``known``) and
#: whether the fit cross-fits.
TRAITS: dict[str, tuple[str, str, bool]] = {
    "binary_logistic": ("balanced", "logistic", False),
    "binary_logistic_crossfit": ("balanced", "logistic", True),
    "binary_logistic_ridge": ("balanced", "ridge", False),
    "binary_logistic_ridge_crossfit": ("balanced", "ridge", True),
    "multinomial_logistic": ("multinomial", "logistic", False),
    "multinomial_logistic_crossfit": ("multinomial", "logistic", True),
    "binary_known": ("balanced", "known", False),
    "binary_unbalanced": ("unbalanced", "logistic", False),
}
PARAMETERS: dict[str, dict[str, tuple[str, str, float, float]]] = {
    scenario: MULTINOMIAL_PARAMETERS if law == "multinomial" else BINARY_PARAMETERS
    for scenario, (law, _, _) in TRAITS.items()
}

#: The scenarios whose mechanism is fitted, so the curve carries a mechanism term.
ESTIMATED = tuple(scenario for scenario, (_, kind, _) in TRAITS.items() if kind != "known")

#: The parameters with a control.  The ATE has none: at a correct model its mechanism term is
#: a reflection of the second-moment term, which keeps the variance (plan section 3.4).
CONTROLLED = ("att", "atc", "att_2v0", "atc_1v0")

#: The truncation bounds that the multinomial and the unbalanced scenarios declare.  The package
#: refuses the plug-in limits (``truncation_active``) when a fitted logit lies within three
#: standard errors of a truncation bound.  The automatic bounds, 0.0229 to 0.025 at ``n = 1000``,
#: sit that close to the extreme rows of the three-arm law and of the unbalanced law in 20 to 55
#: percent of draws, and a refusal rate above the failure bound of 0.01 would make every cell of
#: those scenarios red by construction.  These bounds lie far from every fitted logit, so no
#: cell is truncated and the check does not act, as in the balanced scenarios.  The study covers
#: the untruncated regime only.
WIDE_G_BOUNDS = (0.005, 0.995)
WIDE_G_SCENARIOS = (
    "multinomial_logistic",
    "multinomial_logistic_crossfit",
    "binary_unbalanced",
)

#: The folds and the declared outcome support of every cross-fitted scenario.
CROSSFIT_FOLDS = 5
CROSSFIT_Q_BOUNDS = (-40.0, 45.0)

#: The treatment intercept of ``binary_unbalanced``.
UNBALANCED_INTERCEPT = -1.5


def estimand_key(token: str, row: str) -> str:
    """The primary estimand of one parameter's row: ``plugin_att_lower`` ... ."""
    return f"plugin_{token}_{row}"


def cell_label(scenario: str, token: str, row: str) -> str:
    """The calibration label of one row of one scenario: ``binary_logistic_att_lower`` ... ."""
    return f"{scenario}_{token}_{row}"


def primary_name(scenario: str, token: str) -> str:
    """The primary scenario of one parameter: ``binary_logistic_att`` ... .

    The failure accounting costs every estimand of a failed scenario, so a refusal of one
    parameter loses only that parameter's primary rows when each parameter is its own scenario.
    """
    return f"{scenario}_{token}"


#: Primary scenario name -> (scenario, token).
PRIMARY: dict[str, tuple[str, str]] = {
    primary_name(scenario, token): (scenario, token)
    for scenario, parameters in PARAMETERS.items()
    for token in parameters
}

#: The primary scenarios and their claim estimands.
SCENARIOS: dict[str, tuple[str, ...]] = {
    name: tuple(estimand_key(token, row) for row in CLAIM_ROWS)
    for name, (_, token) in PRIMARY.items()
}

#: The three parameters of a scenario read the sample of the first.
SEED_OWNERS: dict[str, str] = {
    name: primary_name(scenario, next(iter(PARAMETERS[scenario])))
    for name, (scenario, _) in PRIMARY.items()
    if name != primary_name(scenario, next(iter(PARAMETERS[scenario])))
}


def scenario_of(name: str) -> str:
    """The scenario of a primary scenario name, or ``name`` itself when it is a scenario."""
    return PRIMARY[name][0] if name in PRIMARY else name


def control_kind(scenario: str) -> str:
    """The control of a scenario's ATT and ATC rows."""
    return "inflated_se_control" if scenario in ESTIMATED else "shrunken_se_control"


def _cells() -> tuple[str, ...]:
    cells: list[str] = []
    for scenario, parameters in PARAMETERS.items():
        for token in parameters:
            for row in CLAIM_ROWS:
                label = cell_label(scenario, token, row)
                cells.append(f"{label}__correctly_specified")
                if token in CONTROLLED:
                    cells.append(f"{label}__{control_kind(scenario)}")
    return tuple(cells)


PROPERTY_CELLS = {"interval_calibration": _cells()}

#: The unjudged reading of the corrected nu^2: one row for each draw and one for each
#: scenario and parameter.
NU2_ROWS = "nu2-replicates.csv.gz"
NU2_READING = "nu2-reading.csv"

#: The unjudged readings of the property phase.  ``correction-reading.csv`` holds, for each
#: positive cell, the same-draw SE ratio and coverage without the package's variance correction
#: and the mean share the correction removes.  ``refusal-reading.csv`` holds the refusals of
#: each parameter by key.  Neither one carries a verdict.
CORRECTION_READING = "correction-reading.csv"
REFUSAL_READING = "refusal-reading.csv"

STUDY = StudyRecord(
    name="omitted-variable bound plug-in standard error",
    slug="omitted-variable-bound-plugin-se",
    artifacts=ROOT / "tests" / "canonical" / "omitted_variable_bound_plugin",
    document=(
        "docs/technical-reference/method-evidence/omitted-variable-bound-plugin-standard-error.md"
    ),
    anchor="omitted-variable-bound-plug-in-standard-error",
    scenarios=SCENARIOS,
    scenario_seed_owners=SEED_OWNERS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-omitted-variable-bound-plugin",
    reference=None,
    failure_bound={"cleverly-omitted-variable-bound-plugin": 0.01},
    modules=(
        "tests/studies/omitted_variable_bound_plugin.py",
        "tests/studies/omitted_variable_bound_plugin_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.omitted_variable_bound_plugin",
    properties_module="tests.studies.omitted_variable_bound_plugin_properties",
    property_cells=PROPERTY_CELLS,
    extra_artifacts=(NU2_ROWS, NU2_READING),
    property_artifacts=(CORRECTION_READING, REFUSAL_READING),
    publication_policy="reporting",
)

CONFIGURATION: dict[str, Any] = {
    "laws": {
        "balanced": "make_linear_ate(n=1000): W ~ N(0, I_4), g = expit(0.3 W1 - 0.2 W2 + 0.1 W3), "
        "Y = 2 + 1.5 A + W1 + 0.5 W2 - 0.8 W3 + 0.4 W4 + N(0, 1)",
        "unbalanced": "the balanced law with g = expit(-1.5 + 0.3 W1 - 0.2 W2 + 0.1 W3), "
        "treated share about 0.19",
        "multinomial": "W ~ N(0, I_3); logits (0, 0.3 W1 - 0.2 W2, -0.2 W1 + 0.3 W2); "
        "Y = 0.6 step[A] + W1 - 0.5 W2 + 0.2 W3 + N(0, 1), steps (0, 1, 2.4)",
    },
    "fits": {
        "binary_logistic": "in-sample TMLE, LinearRegression outcome, "
        "LogisticRegression(C=inf, max_iter=2000)",
        "binary_logistic_crossfit": "the same with cross_fit=True, n_folds=5, "
        "q_bounds=(-40.0, 45.0)",
        "binary_logistic_ridge": "the same as binary_logistic with "
        "LogisticRegression(max_iter=1000), so C = 1",
        "binary_logistic_ridge_crossfit": "the same as binary_logistic_ridge with "
        "cross_fit=True, n_folds=5, q_bounds=(-40.0, 45.0)",
        "multinomial_logistic": "in-sample TMLE, LinearRegression outcome, "
        f"LogisticRegression(C=inf, max_iter=2000), reference arm 0, g_bounds={WIDE_G_BOUNDS}",
        "multinomial_logistic_crossfit": "the same with cross_fit=True, n_folds=5, "
        f"q_bounds=(-40.0, 45.0), g_bounds={WIDE_G_BOUNDS}",
        "binary_known": "in-sample TMLE, LinearRegression outcome, the law's own mechanism "
        "linear_dgp().propensity declared through treatment_probabilities=",
        "binary_unbalanced": "in-sample TMLE, LinearRegression outcome, "
        "LogisticRegression(C=inf, max_iter=2000), treatment intercept -1.5, "
        f"g_bounds={WIDE_G_BOUNDS}",
    },
    "nu2_estimator": "plugin, second-order corrected by the package (nu2_correction and "
    "nu2_variance_correction are package values)",
    "strength": {"cf_y": CF_Y, "cf_d": CF_D, "rho": RHO},
    "primary_rows": "lower and upper: estimate = the bound, std_error = (lower - ci_lower) / "
    "z_0.95 and (ci_upper - upper) / z_0.95, each checked against the rebuilt curve less the "
    "package variance correction; interval = estimate +/- 1.959964 std_error. One primary "
    "scenario for each parameter, so a refusal of one parameter loses only its rows",
    "correction_reading": "for each positive cell, the same-draw SE ratio and coverage without "
    "the package variance correction, the 99% SE-ratio interval inside the calibration band or "
    "not, and the mean and the largest share of the variance that the correction removes, "
    "published unjudged in correction-reading.csv; the refusals of each parameter by key in "
    "refusal-reading.csv",
    "nu2_reading": "the corrected nu2, its corrected standard error, the raw "
    "E_n[alpha_hat^2], B_hat and the variance correction of every draw, published unjudged in "
    "nu2-replicates.csv.gz and nu2-reading.csv",
    "controls": {
        "inflated_se_control": "the estimated-mechanism scenarios' ATT and ATC bound rows, with "
        "the package curve minus the independently rebuilt mechanism term, with the positive's "
        "relative variance correction (v_control * (v_positive - c^2 v) / v_positive), so the "
        "control's variance is positive whenever the package's is",
        "shrunken_se_control": "binary_known's ATT and ATC bound rows, with the package curve "
        "plus the mechanism term of a study-side LogisticRegression(C=inf) fit, with the "
        "positive's relative variance correction",
    },
    "truth": "sigma2 = 1. balanced: nu2 = 2 + 2 exp(0.07) for ate, 4 exp(0.07) for att and atc, "
    "effect 1.5. unbalanced: 1-D Gauss-Hermite quadrature, effect 1.5. multinomial: 80 x 80 "
    "Gauss-Hermite quadrature, effects 1.44, 1.44, 0.6",
}

#: The red-cell route, declared before any run (X26 plan section 3.5 and review F10, and the
#: pre-queue design review C2 to C5).
#:
#: Owner ids.  Each names the roadmap row ("Red-cell owners") that closes the cells it reads.
RED_CELL_OWNER = "X26-plugin-finite-sample"
CROSSFIT_REFUSAL_OWNER = "X26-crossfit-refusals"
UNDERPOWERED_CONTROL_OWNER = "X26-unbalanced-control"
CORRECTION_OVERSHOOT_OWNER = "X26-correction-overshoot"
REFUSAL_LIMITED_OWNER = "X26-refusal-limited"

#: The declarations that the owner of the study has not yet confirmed.  The working decision keeps
#: the package's variance correction of :math:`\hat\nu^2` in the estimator.  It rests on the
#: pilot (``_x26/pilot``).  On the pilot's draws the correction moves the unbalanced ATC SE ratio
#: from 1.039 to 0.951.  Two things follow, and each is declared here and nowhere else, so a
#: different decision changes these constants and :data:`RED_CELL_POLICY` only.
#:
#: * :data:`EXPECTED_BOUND_EXCEEDED` lists the scenarios whose refusal rate the pilot put above
#:   the failure bound (``FAILURE_BOUND`` in the properties module stays 0.01).  Their cells read
#:   ``bound exceeded`` and are published red with the refusal rate by key.
#: * :data:`UNDERPOWERED_CONTROLS` lists the controls that the design test already predicts as
#:   weak and that the pilot read inside the band (1.052 and 1.053 against a design ratio of
#:   1.11).  They read "control underpowered".
EXPECTED_BOUND_EXCEEDED = ("multinomial_logistic_crossfit",)
UNDERPOWERED_CONTROLS = (
    "binary_unbalanced_atc_lower__inflated_se_control",
    "binary_unbalanced_atc_upper__inflated_se_control",
)


#: The key a refusal of the package names.  The study raises :class:`DataError` for these three
#: and for no other reason.
REFUSAL_KEYS = ("truncation_active", "correction_unstable", "corrected_nonpositive")


def refusal_key(message: str) -> str:
    """The refusal key a failure record's message names, or ``other``.

    The study raises ``DataError("the plug-in bound of 'name' refused: key")`` for the three
    refusals of the package (:data:`REFUSAL_KEYS`).  Any other recorded failure, such as a
    failure of the fit itself, reads ``other``.
    """
    for key in REFUSAL_KEYS:
        if message.endswith(f"refused: {key}"):
            return key
    return "other"


def cell_parameter(cell: str) -> tuple[str, str]:
    """The scenario and the parameter token whose fit fills a property cell."""
    label = cell.split("__")[0]
    for scenario, parameters in PARAMETERS.items():
        for token in parameters:
            if label in {cell_label(scenario, token, row) for row in CLAIM_ROWS}:
                return scenario, token
    raise KeyError(f"{STUDY.slug} has no property cell {cell!r}")


def expected_bound_exceeded_cells() -> tuple[str, ...]:
    """The property cells that the policy declares to read ``bound exceeded``."""
    return tuple(
        cell
        for cell in PROPERTY_CELLS["interval_calibration"]
        if cell_parameter(cell)[0] in EXPECTED_BOUND_EXCEEDED
    )


def expected_bound_exceeded_primary() -> tuple[str, ...]:
    """The primary scenarios that the policy declares to read ``bound exceeded``."""
    return tuple(
        name for name, (scenario, _) in PRIMARY.items() if scenario in EXPECTED_BOUND_EXCEEDED
    )


_EXPECTED_CELLS = expected_bound_exceeded_cells()

RED_CELL_POLICY = (
    "the study publishes under reporting; no replicate count, margin, law, learner or "
    "strength changes after a verdict is seen",
    "a red positive cell is diagnosed first: on 20 replicates of its scenario, the "
    "infinitesimal jackknife (unit witness W5) and the numerical sandwich "
    "(TestTheNumericalSandwich) are compared with the package curve. The diagnosis also reads "
    f"{CORRECTION_READING}: the same-draw SE ratio without the package's variance correction, "
    "and the mean share of the variance that the correction removes. A mismatch of the curve is "
    "a defect, which is fixed and the study regenerated under a fresh declaration commit",
    "when they agree, the cell is rerun at n = 4000 with 5,000 replicates and published red "
    f"with owner {RED_CELL_OWNER} and the reading 'finite-sample, contracting' or "
    "'finite-sample, not contracting at n=4000', with a roadmap owner row",
    "a red positive cell reads as finite-sample only when its 99% SE-ratio interval contains "
    "1, its coverage is consistent with 0.95, and the diagnosis finds no defect; any other "
    "red positive cell is a defect suspect and never ends as an owner row unfixed, except a "
    "red cell that rule 7, 8 or 9 reads",
    "a control that does not leave the band is reported as 'control underpowered', with its "
    f"ratio. {len(UNDERPOWERED_CONTROLS)} controls ({', '.join(UNDERPOWERED_CONTROLS)}) are "
    f"declared so before the run, with owner {UNDERPOWERED_CONTROL_OWNER}: the pilot read the "
    "ratios of the ATC controls at 1.052 and 1.053 against a design ratio of 1.11",
    "the corrected nu2 rows are measured, not claimed: their reading is never a verdict, and a "
    "shortfall in it opens no red cell",
    f"the cells of {', '.join(EXPECTED_BOUND_EXCEEDED)} ({len(_EXPECTED_CELLS)} property cells and "
    f"{len(expected_bound_exceeded_primary())} primary scenarios) are declared to read 'bound "
    "exceeded', because the "
    "pilot refused 1.7 to 4.2 percent of its draws and the failure bound stays 0.01. They are "
    f"published red with owner {CROSSFIT_REFUSAL_OWNER}, with the refusal rate of each key in "
    f"{REFUSAL_READING} and with the complete-case SE ratio and coverage labelled 'not a "
    "verdict'. Rules 2 to 4 do not apply to a cell that only the bound turns red",
    "a cell that is green on the complete case and red only because a worst-case clause fails "
    f"reads 'refusal-limited' with owner {REFUSAL_LIMITED_OWNER}. It publishes the number of "
    "lost replicates, the complete-case value and the worst-case value. It is not a defect "
    "suspect under rule 4",
    "a red positive cell whose 99% SE-ratio interval without the variance correction lies "
    f"inside the calibration band ({CORRECTION_READING}, column uncorrected_in_band), and whose "
    "diagnosis under rule 2 finds no mismatch of the curve, reads 'variance-correction "
    f"overshoot' with owner {CORRECTION_OVERSHOOT_OWNER}. It is rerun under rule 3 and "
    "published as 'contracting' or 'not contracting at n=4000'. It is not a defect suspect "
    "under rule 4",
)

CONFIGURATION["red_cell_policy"] = list(RED_CELL_POLICY)
CONFIGURATION["declared_red_cells"] = {
    "bound_exceeded": {
        "scenarios": list(EXPECTED_BOUND_EXCEEDED),
        "property_cells": list(_EXPECTED_CELLS),
        "primary_scenarios": list(expected_bound_exceeded_primary()),
        "owner": CROSSFIT_REFUSAL_OWNER,
    },
    "control_underpowered": {
        "cells": list(UNDERPOWERED_CONTROLS),
        "owner": UNDERPOWERED_CONTROL_OWNER,
    },
    "refusal_limited": {"owner": REFUSAL_LIMITED_OWNER},
    "correction_overshoot": {"owner": CORRECTION_OVERSHOOT_OWNER},
    "finite_sample": {"owner": RED_CELL_OWNER},
}

#: The two-sided 95% multiplier of each row's interval, and the one-sided multiplier the bound
#: reports its limit with.
TWO_SIDED = float(stats.norm.ppf(0.975))
ONE_SIDED = float(stats.norm.ppf(0.95))

#: The variance of the linear index of the binary mechanism, ``0.3^2 + 0.2^2 + 0.1^2``.
INDEX_VARIANCE = 0.14

#: The three-arm law's arm steps and outcome coefficients.
MULTINOMIAL_STEPS = (0.0, 1.0, 2.4)
MULTINOMIAL_EFFECT_SCALE = 0.6

#: The nodes per axis of the quadrature behind the truths that have no closed form.
QUADRATURE_NODES = 80


def strength() -> float:
    """The multiplier on the bias scale, ``|rho| sqrt(cf_y cf_d / (1 - cf_d))``."""
    return abs(RHO) * math.sqrt(CF_Y * CF_D / (1.0 - CF_D))


def multinomial_logits(w: np.ndarray) -> np.ndarray:
    """``(n, 3)`` logits of the three-arm law, arm 0 first."""
    return np.column_stack(
        [np.zeros(w.shape[0]), 0.3 * w[:, 0] - 0.2 * w[:, 1], -0.2 * w[:, 0] + 0.3 * w[:, 1]]
    )


def multinomial_mechanism(w: np.ndarray) -> np.ndarray:
    """``(n, 3)`` treatment mechanism of the three-arm law."""
    eta = multinomial_logits(w)
    eta = eta - eta.max(axis=1, keepdims=True)
    expd = np.exp(eta)
    return np.asarray(expd / expd.sum(axis=1, keepdims=True), dtype=float)


def multinomial_second_moments(nodes: int = QUADRATURE_NODES) -> dict[str, float]:
    """The three-arm ``nu^2`` of each parameter, by Gauss-Hermite quadrature on ``(W1, W2)``.

    The mechanism reads ``W1`` and ``W2`` only, so a product rule on those two axes is exact
    up to the rule's error.
    """
    x, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    w1, w2 = np.meshgrid(x, x, indexing="ij")
    omega = np.outer(weights, weights).reshape(-1)
    g = multinomial_mechanism(np.column_stack([w1.reshape(-1), w2.reshape(-1)]))
    share = omega @ g
    return {
        "ate_2v0": float(omega @ (1.0 / g[:, 2] + 1.0 / g[:, 0])),
        "att_2v0": float((share[2] + omega @ (g[:, 2] ** 2 / g[:, 0])) / share[2] ** 2),
        "atc_1v0": float((share[0] + omega @ (g[:, 0] ** 2 / g[:, 1])) / share[0] ** 2),
    }


def binary_second_moments(
    intercept: float, nodes: int = QUADRATURE_NODES
) -> tuple[dict[str, float], float]:
    r"""The binary ``nu^2`` of each parameter and the treated share, by quadrature over the index.

    The index :math:`\eta = b + 0.3 W_1 - 0.2 W_2 + 0.1 W_3` is normal with mean ``intercept``
    and variance :data:`INDEX_VARIANCE`, so a one-dimensional Gauss-Hermite rule integrates
    every function of :math:`g = \mathrm{expit}(\eta)`.
    """
    x, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    g = expit(intercept + math.sqrt(INDEX_VARIANCE) * x)
    share = float(weights @ g)
    return (
        {
            "ate": float(weights @ (1.0 / g + 1.0 / (1.0 - g))),
            "att": float((share + weights @ (g**2 / (1.0 - g))) / share**2),
            "atc": float(((1.0 - share) + weights @ ((1.0 - g) ** 2 / g)) / (1.0 - share) ** 2),
        },
        share,
    )


def _balanced_second_moments() -> dict[str, float]:
    odds = math.exp(INDEX_VARIANCE / 2.0)
    return {"ate": 2.0 + 2.0 * odds, "att": 4.0 * odds, "atc": 4.0 * odds}


_MULTINOMIAL_NU2 = multinomial_second_moments()
_UNBALANCED_NU2 = binary_second_moments(UNBALANCED_INTERCEPT)[0]


def truths(scenario: str) -> dict[str, float]:
    """The population ``nu^2`` and both bounds of each parameter of ``scenario``."""
    scenario = scenario_of(scenario)
    if scenario not in PARAMETERS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    law = TRAITS[scenario][0]
    if law == "multinomial":
        nu2 = _MULTINOMIAL_NU2
        effects = {
            "ate_2v0": MULTINOMIAL_EFFECT_SCALE * MULTINOMIAL_STEPS[2],
            "att_2v0": MULTINOMIAL_EFFECT_SCALE * MULTINOMIAL_STEPS[2],
            "atc_1v0": MULTINOMIAL_EFFECT_SCALE * MULTINOMIAL_STEPS[1],
        }
    else:
        nu2 = _UNBALANCED_NU2 if law == "unbalanced" else _balanced_second_moments()
        effects = dict.fromkeys(BINARY_PARAMETERS, 1.5)
    sigma2 = 1.0
    values: dict[str, float] = {}
    for token in PARAMETERS[scenario]:
        bias = strength() * math.sqrt(sigma2 * nu2[token])
        values[estimand_key(token, "nu2")] = nu2[token]
        values[estimand_key(token, "lower")] = effects[token] - bias
        values[estimand_key(token, "upper")] = effects[token] + bias
    return values


def draw_multinomial(n: int, seed: int) -> pd.DataFrame:
    """One sample of the three-arm law, columns ``Y, A, W1, W2, W3``."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 3))
    g = multinomial_mechanism(w)
    arm = (rng.random(n)[:, None] > np.cumsum(g, axis=1)).sum(axis=1)
    arm = np.minimum(arm, 2)
    steps = np.asarray(MULTINOMIAL_STEPS)[arm]
    outcome = (
        MULTINOMIAL_EFFECT_SCALE * steps
        + w[:, 0]
        - 0.5 * w[:, 1]
        + 0.2 * w[:, 2]
        + rng.standard_normal(n)
    )
    frame = pd.DataFrame(w, columns=["W1", "W2", "W3"])
    frame.insert(0, "A", arm.astype(float))
    frame.insert(0, "Y", outcome)
    return frame


def draw_unbalanced(n: int, seed: int) -> pd.DataFrame:
    """One sample of the unbalanced binary law, columns ``Y, A, W1, W2, W3, W4``.

    The law of :func:`cleverly.datasets.make_linear_ate` with the treatment intercept moved to
    :data:`UNBALANCED_INTERCEPT`.  The draws follow that generator's order: the covariates, the
    treatment, then the outcome noise.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 4))
    g = expit(UNBALANCED_INTERCEPT + 0.3 * w[:, 0] - 0.2 * w[:, 1] + 0.1 * w[:, 2])
    a = rng.binomial(1, g).astype(float)
    y = 2.0 + 1.5 * a + w[:, 0] + 0.5 * w[:, 1] - 0.8 * w[:, 2] + 0.4 * w[:, 3]
    y = y + rng.normal(size=n)
    frame = pd.DataFrame(w, columns=["W1", "W2", "W3", "W4"])
    frame.insert(0, "A", a)
    frame.insert(0, "Y", y)
    return frame


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit.

    ``scenario`` is a scenario or a primary scenario name.
    """
    scenario = scenario_of(scenario)
    truth = truths(scenario)
    law = TRAITS[scenario][0]
    if law == "multinomial":
        return draw_multinomial(n, seed), truth
    if law == "unbalanced":
        return draw_unbalanced(n, seed), truth
    frame, _ = make_linear_ate(n=n, seed=seed)
    return frame, truth


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def known_mechanism(frame: pd.DataFrame) -> np.ndarray:
    """``(n,)`` ``P(A = 1 | W)`` of ``make_linear_ate``, the law's own mechanism."""
    covariates = frame.loc[:, ["W1", "W2", "W3", "W4"]].to_numpy(dtype=float)
    return np.asarray(linear_dgp().propensity(covariates), dtype=float)


def _treatment_learner(scenario: str) -> LogisticRegression:
    if TRAITS[scenario][1] == "ridge":
        return LogisticRegression(max_iter=1000)
    return LogisticRegression(C=np.inf, max_iter=2000)


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The declared TMLE of ``scenario``.

    No warning is suppressed.  The declared fits raise none on these laws, so a warning that a
    later package raises reaches the regeneration log instead of vanishing.
    """
    scenario = scenario_of(scenario)
    estimands: tuple[Literal["ate", "att", "atc"], ...] = ("ate", "att", "atc")
    if TRAITS[scenario][1] == "known":
        return (
            TMLE(
                outcome_learner=LinearRegression(),
                cross_fit=False,
                simultaneous=False,
                estimands=estimands,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                treatment_probabilities=known_mechanism(frame),
            )
            .single()
        )
    crossfit = TRAITS[scenario][2]
    options: dict[str, Any] = {"cross_fit": crossfit}
    if crossfit:
        options.update(n_folds=CROSSFIT_FOLDS, q_bounds=CROSSFIT_Q_BOUNDS)
    if scenario in WIDE_G_SCENARIOS:
        options["g_bounds"] = WIDE_G_BOUNDS
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=_treatment_learner(scenario),
            simultaneous=False,
            estimands=estimands,
            random_state=0,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


# ---------------------------------------------------------------- the independent rebuild

#: How closely the rebuilt curve, mechanism and representer must reproduce the package's
#: own, relative to the largest absolute value of the package's array (and at least 1).
REBUILD_TOLERANCE = 1e-8


def _close(rebuilt: np.ndarray, reported: np.ndarray) -> bool:
    scale = max(1.0, float(np.max(np.abs(reported))))
    return bool(np.max(np.abs(rebuilt - reported)) <= REBUILD_TOLERANCE * scale)


def _softmax_reference(design: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """``(n, K)`` probabilities with the reference arm first, from reference-form ``theta``."""
    eta = np.column_stack([np.zeros(design.shape[0]), design @ theta.T])
    eta = eta - eta.max(axis=1, keepdims=True)
    expd = np.exp(eta)
    return np.asarray(expd / expd.sum(axis=1, keepdims=True), dtype=float)


@dataclass(frozen=True)
class Logistic:
    """One fold's fitted logistic mechanism in reference form.

    Parameters
    ----------
    design : ndarray
        ``(n, d)`` intercept and covariates of every row.
    theta : ndarray
        ``(K - 1, d)`` coefficients against the lowest arm.
    penalty : ndarray
        ``(p, p)`` matrix ``P / C`` with ``p = (K - 1) d``.
    train : ndarray
        The rows the fold's model was fitted on.
    test : ndarray
        The rows the fold's model predicts.
    """

    design: np.ndarray
    theta: np.ndarray
    penalty: np.ndarray
    train: np.ndarray
    test: np.ndarray


def _penalty(c: float, arms: int, d: int) -> np.ndarray:
    """scikit-learn's L2 penalty in reference form, the intercept unpenalized."""
    p = (arms - 1) * d
    if math.isinf(c):
        return np.zeros((p, p))
    slopes = np.eye(d)
    slopes[0, 0] = 0.0
    between = np.ones((1, 1)) if arms == 2 else np.eye(arms - 1) - 1.0 / arms
    return np.asarray(np.kron(between, slopes) / c, dtype=float)


def _reference_theta(coef: np.ndarray, intercept: np.ndarray, arms: int) -> np.ndarray:
    full = np.column_stack([np.asarray(intercept).reshape(-1), np.asarray(coef)])
    if arms == 2:
        return np.asarray(full[:1], dtype=float)
    return np.asarray(full[1:] - full[0], dtype=float)


def _fold_rows(folds: Any, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds.n_folds == 1:
        everything = np.arange(n)
        return [(everything, everything)]
    return [(np.asarray(train), np.asarray(test)) for train, test in folds]


def captured_logistics(result: Any) -> list[Logistic]:
    """Each fold's mechanism, from the coefficients the fit captured, in fold order."""
    nuisance = result.repeats[0].nuisance
    arms = len(nuisance.arms)
    covariates = np.asarray(result.data.covariates, dtype=float)
    design = np.column_stack([np.ones(covariates.shape[0]), covariates])
    rows = _fold_rows(nuisance.folds, result.n)
    fits = nuisance.treatment_fits
    if fits is None or len(fits) != len(rows):
        raise RuntimeError("the fit captured no logistic mechanism for each fold")
    out = []
    for fit_record, (train, test) in zip(fits, rows, strict=True):
        if fit_record.steps or not fit_record.fit_intercept:
            raise RuntimeError("the declared mechanism is a bare LogisticRegression")
        theta = _reference_theta(fit_record.coef, fit_record.intercept, arms)
        out.append(
            Logistic(
                design=design,
                theta=theta,
                penalty=_penalty(float(fit_record.C), arms, design.shape[1]),
                train=train,
                test=test,
            )
        )
    return out


def study_logistic(result: Any) -> list[Logistic]:
    """A study-side in-sample ``LogisticRegression(C=inf)`` mechanism, for the known fits."""
    covariates = np.asarray(result.data.covariates, dtype=float)
    treatment = np.asarray(result.data.treatment, dtype=float)
    model = LogisticRegression(C=np.inf, max_iter=2000).fit(covariates, treatment)
    arms = len(result.repeats[0].nuisance.arms)
    everything = np.arange(covariates.shape[0])
    return [
        Logistic(
            design=np.column_stack([np.ones(covariates.shape[0]), covariates]),
            theta=_reference_theta(model.coef_, model.intercept_, arms),
            penalty=_penalty(math.inf, arms, covariates.shape[1] + 1),
            train=everything,
            test=everything,
        )
    ]


def _truncate(g: np.ndarray, bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The truncated mechanism and the cells it moved, by the package's documented rule.

    Two arms clip ``g_1`` and take ``g_0 = 1 - g_1``, marking both columns of a moved row.
    More arms clip column by column without renormalizing.
    """
    lower, upper = bounds
    if g.shape[1] == 2:
        one = np.clip(g[:, 1], lower, upper)
        moved = one != g[:, 1]
        return np.column_stack([1.0 - one, one]), np.column_stack([moved, moved])
    clipped = np.clip(g, lower, upper)
    return clipped, clipped != g


def representer(
    group: str, arm: int, versus: int, treatment: np.ndarray, g: np.ndarray, share: float | None
) -> tuple[np.ndarray, np.ndarray]:
    r"""``alpha`` at each row and its derivative in each column of ``g``.

    Columns are arm indices.  ``mean`` is the contrast :math:`1\{A=a\}/g_a - 1\{A=r\}/g_r`;
    ``att`` is :math:`(1\{A=a\} - 1\{A=r\} g_a / g_r) / p_a`; ``atc`` is
    :math:`(1\{A=a\} g_r / g_a - 1\{A=r\}) / p_r`.
    """
    on_a = (treatment == arm).astype(float)
    on_r = (treatment == versus).astype(float)
    derivative = np.zeros_like(g)
    if group == "mean":
        alpha = on_a / g[:, arm] - on_r / g[:, versus]
        derivative[:, arm] = -on_a / g[:, arm] ** 2
        derivative[:, versus] = on_r / g[:, versus] ** 2
        return alpha, derivative
    assert share is not None
    if group == "att":
        alpha = (on_a - on_r * g[:, arm] / g[:, versus]) / share
        derivative[:, arm] = -on_r / (g[:, versus] * share)
        derivative[:, versus] = on_r * g[:, arm] / (g[:, versus] ** 2 * share)
        return alpha, derivative
    alpha = (on_a * g[:, versus] / g[:, arm] - on_r) / share
    derivative[:, versus] = on_a / (g[:, arm] * share)
    derivative[:, arm] = -on_a * g[:, versus] / (g[:, arm] ** 2 * share)
    return alpha, derivative


def mechanism_term(
    folds: list[Logistic],
    treatment_index: np.ndarray,
    weights: np.ndarray,
    alpha: np.ndarray,
    derivative: np.ndarray,
) -> np.ndarray:
    r"""The stacked mechanism term :math:`\psi^\gamma` of :math:`E_n[w \alpha^2]`.

    For fold :math:`k`, :math:`J_k = \sum_{\mathrm{train}} w [\mathrm{diag}(g_{-r}) -
    g_{-r} g_{-r}^\top] \otimes x x^\top + P/C`,
    :math:`D_k = n^{-1} \sum_{\mathrm{test}} 2 w \alpha\, \partial_\theta \alpha`, and the row
    function is :math:`u_{k,i} = w_i x_i \otimes (e_{A_i} - g_k)_{-r} 1\{i \in
    \mathrm{train}_k\} - (w_i/n) P\theta_k/C`.  The term is
    :math:`\psi^\gamma_i = n \sum_k D_k^\top J_k^{-1} u_{k,i}`.  ``derivative`` is already zero
    on every truncated cell.
    """
    n = treatment_index.shape[0]
    term = np.zeros(n)
    for fold in folds:
        g = _softmax_reference(fold.design, fold.theta)
        arms = g.shape[1]
        d = fold.design.shape[1]
        reduced = g[:, 1:]
        information = np.zeros(((arms - 1) * d, (arms - 1) * d))
        train = fold.train
        x_train = fold.design[train]
        for m in range(arms - 1):
            for q in range(arms - 1):
                factor = reduced[train, m] * (float(m == q) - reduced[train, q])
                block = (x_train * (weights[train] * factor)[:, None]).T @ x_train
                information[m * d : (m + 1) * d, q * d : (q + 1) * d] = block
        information = information + fold.penalty
        # d alpha / d theta_m = x g_m (b_m - sum_c b_c g_c), reference column dropped.
        inner = (derivative * g).sum(axis=1)
        chain = g[:, 1:] * (derivative[:, 1:] - inner[:, None])
        test = fold.test
        coefficient = (2.0 * weights[test] * alpha[test])[:, None] * chain[test]
        gradient = np.einsum("nm,nd->md", coefficient, fold.design[test]).reshape(-1) / n
        direction = np.linalg.solve(information, gradient)
        indicator = np.zeros_like(g)
        indicator[np.arange(n), treatment_index] = 1.0
        residual = (indicator - g)[:, 1:]
        score = np.einsum("nm,nd->nmd", residual, fold.design).reshape(n, -1)
        term[train] += n * weights[train] * (score[train] @ direction)
        term -= weights * float(direction @ (fold.penalty @ fold.theta.reshape(-1)))
    return term


@dataclass(frozen=True)
class Rebuilt:
    """One parameter's curve of ``nu^2``, rebuilt without the package's curve.

    Parameters
    ----------
    nu2 : float
        The uncorrected plug-in :math:`E_n[\\hat\\alpha^2]`.
    second_moment : ndarray
        ``w (alpha^2 - nu^2)``.
    share_term : ndarray
        The influence of the conditioning share, zero for the ATE.
    mechanism_term : ndarray
        The stacked mechanism term; zero for a declared known mechanism.
    """

    nu2: float
    second_moment: np.ndarray
    share_term: np.ndarray
    mechanism_term: np.ndarray

    @property
    def curve(self) -> np.ndarray:
        return self.second_moment + self.share_term + self.mechanism_term


def rebuild(result: Any, scenario: str, token: str, elements: Any) -> Rebuilt:
    """Rebuild the plug-in curve of one parameter and check it against the package's.

    The mechanism comes from the captured coefficients (an estimated mechanism) or from the
    law (a known one).  The representer, the uncorrected ``nu^2``, the share term and the
    mechanism term are this module's.  The package reports ``nu^2`` less its second-order
    correction, so the rebuilt value is compared with the reported value plus that correction.
    The correction itself is not rebuilt.  Any departure beyond :data:`REBUILD_TOLERANCE`
    raises ``RuntimeError`` from this module, which stops the run: a departure is a defect, not
    a property of a sample.
    """
    _, group, arm_code, versus_code = PARAMETERS[scenario][token]
    nuisance = result.repeats[0].nuisance
    arms = tuple(float(a) for a in nuisance.arms)
    arm, versus = arms.index(arm_code), arms.index(versus_code)
    data = result.data
    n = result.n
    weights = np.asarray(data.weights, dtype=float)
    treatment_index = np.searchsorted(np.asarray(arms), np.asarray(data.treatment, dtype=float))
    stored = np.asarray(nuisance.propensity.values, dtype=float)
    folds: list[Logistic] = []
    if TRAITS[scenario][1] == "known":
        one = known_mechanism(_frame_of(result))
        g = np.column_stack([1.0 - one, one])
    else:
        folds = captured_logistics(result)
        g = np.zeros((n, len(arms)))
        for fold in folds:
            g[fold.test] = _softmax_reference(fold.design, fold.theta)[fold.test]
    if not _close(g, stored):
        raise RuntimeError(f"the rebuilt mechanism of {scenario!r} is not the package's")
    bounds = (
        result.config.g_bounds_conditional if group in ("att", "atc") else result.config.g_bounds
    )
    bounded, moved = _truncate(g, (float(bounds[0]), float(bounds[1])))
    conditioning = {"att": arm, "atc": versus}.get(group)
    share = None
    indicator = np.zeros(n)
    if conditioning is not None:
        indicator = (treatment_index == conditioning).astype(float)
        share = float(np.average(indicator, weights=weights))
    alpha, derivative = representer(group, arm, versus, treatment_index, bounded, share)
    if not _close(alpha, np.asarray(elements.riesz_representer, dtype=float)):
        raise RuntimeError(f"the rebuilt representer of {token!r} is not the package's")
    nu2 = float(np.average(alpha**2, weights=weights))
    reported_raw = float(elements.nu2) + float(elements.nu2_correction)
    if abs(nu2 - reported_raw) > REBUILD_TOLERANCE * max(1.0, abs(reported_raw)):
        raise RuntimeError(f"the rebuilt nu^2 of {token!r} is not the package's")
    share_term = np.zeros(n)
    if share is not None:
        share_term = -2.0 * nu2 * weights * (indicator - share) / share
    term = (
        mechanism_term(folds, treatment_index, weights, alpha, np.where(moved, 0.0, derivative))
        if folds
        else np.zeros(n)
    )
    rebuilt = Rebuilt(
        nu2=nu2,
        second_moment=weights * (alpha**2 - nu2),
        share_term=share_term,
        mechanism_term=term,
    )
    assert elements.psi_nu2 is not None
    if not _close(rebuilt.curve, np.asarray(elements.psi_nu2, dtype=float)):
        raise RuntimeError(f"the rebuilt curve of nu^2 of {token!r} is not the package's")
    return rebuilt


def _frame_of(result: Any) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(result.data.covariates, dtype=float), columns=list(result.data.covariate_names)
    )


def study_mechanism_term(result: Any, scenario: str, token: str) -> np.ndarray:
    r"""The mechanism term a study-side logistic fit adds to a known-mechanism curve.

    The representer and its derivative are the declared mechanism's; the score, the
    information and :math:`\partial g / \partial\theta` are those of an in-sample
    ``LogisticRegression(C=inf)`` on the same covariates.  This is the term the package would
    add if it treated the declared mechanism as fitted.
    """
    _, group, arm_code, versus_code = PARAMETERS[scenario][token]
    arms = tuple(float(a) for a in result.repeats[0].nuisance.arms)
    arm, versus = arms.index(arm_code), arms.index(versus_code)
    data = result.data
    weights = np.asarray(data.weights, dtype=float)
    treatment_index = np.searchsorted(np.asarray(arms), np.asarray(data.treatment, dtype=float))
    one = known_mechanism(_frame_of(result))
    g0 = np.column_stack([1.0 - one, one])
    conditioning = {"att": arm, "atc": versus}.get(group)
    share = None
    if conditioning is not None:
        share = float(np.average(treatment_index == conditioning, weights=weights))
    alpha, derivative = representer(group, arm, versus, treatment_index, g0, share)
    return mechanism_term(study_logistic(result), treatment_index, weights, alpha, derivative)


# ---------------------------------------------------------------- the rows


def _bias_curve(elements: Any, psi_nu2: np.ndarray) -> np.ndarray:
    """Theorem 4's curve of the maximal bias, from a curve of ``nu^2``.

    ``elements.nu2`` and ``elements.max_bias`` are the corrected values the package reports.
    """
    return np.asarray(
        (elements.sigma2 * psi_nu2 + elements.nu2 * np.asarray(elements.psi_sigma2))
        / (2.0 * elements.max_bias),
        dtype=float,
    )


def _corrected_variance(raw: float, correction: float, what: str) -> float:
    """A variance less a package variance correction.

    The package returned limits before this runs, and it applies the same subtraction to the
    same curve.  A variance that is not positive here is a defect of the study's rebuild and
    stops the run.  It is not a refusal of the package, so it is never a recorded replicate.
    """
    value = raw - correction
    if not value > 0.0:
        raise RuntimeError(
            f"the rebuilt corrected variance of {what} is {value:.6g}, though the package "
            "returned limits"
        )
    return value


def _nu2_standard_error(result: Any, elements: Any, psi_nu2: np.ndarray) -> float:
    """The corrected standard error of ``nu^2`` from one curve of ``nu^2``."""
    raw = float(influence_variance(psi_nu2, result.data.cluster))
    return math.sqrt(_corrected_variance(raw, float(elements.nu2_variance_correction), "nu^2"))


@dataclass(frozen=True)
class BoundVariances:
    r"""The variance of each bound end before the package's variance correction.

    Parameters
    ----------
    raw : dict
        The variance of the combined curve of ``lower`` and of ``upper``, before the correction.
    correction : float
        :math:`c^2 v`, with :math:`v` the package's variance correction of :math:`\hat\nu^2` and
        :math:`c = \kappa \sigma^2 / (2 \sqrt{\sigma^2 \nu^2})` the delta-method coefficient
        of the bound in :math:`\nu^2`.  The coefficient is this module's.
    """

    raw: dict[str, float]
    correction: float

    def standard_error(self, row: str, what: str) -> float:
        """The standard error of one bound end, less the correction."""
        return math.sqrt(_corrected_variance(self.raw[row], self.correction, f"{what} {row}"))

    def uncorrected_standard_error(self, row: str) -> float:
        """The standard error of one bound end without the correction."""
        return math.sqrt(self.raw[row])

    def share(self, row: str) -> float:
        """The share of one bound end's variance that the correction removes."""
        return self.correction / self.raw[row]


def _bound_variances(
    result: Any, name: str, elements: Any, psi_nu2: np.ndarray, strength_value: float
) -> BoundVariances:
    """The variance of each bound end, from one curve of ``nu^2``, before the correction."""
    cluster = result.data.cluster
    curve = np.asarray(result[name].influence_curve, dtype=float)
    bias = _bias_curve(elements, psi_nu2)
    coefficient = strength_value * float(elements.sigma2) / (2.0 * float(elements.max_bias))
    return BoundVariances(
        raw={
            "lower": float(influence_variance(curve - strength_value * bias, cluster)),
            "upper": float(influence_variance(curve + strength_value * bias, cluster)),
        },
        correction=coefficient**2 * float(elements.nu2_variance_correction),
    )


def _control_standard_errors(positive: BoundVariances, control: BoundVariances) -> dict[str, float]:
    """A control's standard error, with the positive's *relative* variance correction.

    A control withholds or adds a term, so its raw variance differs from the positive's.  The
    package's absolute correction is a quantity of the positive's curve.  Taken from a smaller
    control curve it can leave a variance below zero while the package returned limits (two of
    the six unbalanced ATC losses in the pilot).  The control therefore loses the same share of
    its variance as the positive does: ``v_control (v_positive - c^2 v) / v_positive``.  That is
    positive whenever the positive's corrected variance is, so a control never fails a replicate
    the package returned.
    """
    return {row: math.sqrt(control.raw[row] * (1.0 - positive.share(row))) for row in CLAIM_ROWS}


@dataclass(frozen=True)
class Bound:
    """One bound end of one draw.

    Parameters
    ----------
    estimate : float
        The reported limit.
    std_error : float
        The standard error read back off the reported confidence limit.
    control : float or None
        The control's standard error, or ``None`` for a parameter without one.
    std_error_uncorrected : float
        The standard error of the same draw without the package's variance correction.
    correction_share : float
        The share of the bound's variance that the package's variance correction removes on
        this draw, so ``std_error`` is ``std_error_uncorrected * sqrt(1 - correction_share)``.
    """

    estimate: float
    std_error: float
    control: float | None
    std_error_uncorrected: float
    correction_share: float


@dataclass(frozen=True)
class Nu2:
    """The corrected ``nu^2`` of one draw, and what it was corrected from.

    Parameters
    ----------
    estimate : float
        The reported ``nu^2``, :math:`E_n[\\hat\\alpha^2] - \\hat B`.
    std_error : float
        The corrected standard error.
    raw_estimate : float
        :math:`E_n[\\hat\\alpha^2]`.
    raw_std_error : float
        The standard error of the uncorrected curve.
    correction : float
        :math:`\\hat B`, a package value.
    variance_correction : float
        The package's second-order bias of the variance of :math:`\\hat\\nu^2`.
    """

    estimate: float
    std_error: float
    raw_estimate: float
    raw_std_error: float
    correction: float
    variance_correction: float


@dataclass(frozen=True)
class ParameterValues:
    """Everything one fit gives for one parameter.

    Parameters
    ----------
    bounds : dict
        The :class:`Bound` of ``lower`` and of ``upper``.
    nu2 : Nu2
        The unjudged reading.
    """

    bounds: dict[str, Bound]
    nu2: Nu2


def _refusal_key_of(refusal: CapabilityError) -> str | None:
    """The key of the package's variance refusal, or ``None`` when it is another refusal."""
    return "corrected_nonpositive" if "(corrected_nonpositive)" in str(refusal) else None


def parameter_rows(result: Any, scenario: str, token: str) -> ParameterValues:
    """Each row's estimate, reported standard error and control standard error.

    The reported standard error of a bound end is read back off its limit, and it must equal
    the standard error of the rebuilt curve less the package's variance correction.  The
    control is ``None`` for a parameter without one.  Each bound also carries the same draw's
    standard error without the correction and the share the correction removes, which the
    property study publishes as the correction reading.

    Raises
    ------
    DataError
        If the package refuses the limits of this fit (``truncation_active``,
        ``correction_unstable`` or ``corrected_nonpositive``).  The driver records it as a
        failed replicate of this parameter, and of no other.  A refusal of the bound that is not
        ``corrected_nonpositive``, and a variance of the study's own rebuild that is not
        positive, are not recorded: they stop the run.
    """
    ov = omitted_variable_module
    scenario = scenario_of(scenario)
    name = PARAMETERS[scenario][token][0]
    elements = ov.sensitivity_elements(result, name, nu2_estimator="plugin")
    expected = "known" if TRAITS[scenario][1] == "known" else "logistic"
    if elements.limit_refusal is not None:
        raise DataError(f"the plug-in bound of {name!r} refused: {elements.limit_refusal.key}")
    if (
        elements.plugin_mechanism != expected
        or elements.psi_nu2 is None
        or elements.nu2_correction is None
        or elements.nu2_variance_correction is None
    ):
        raise RuntimeError(f"the plug-in bound of {name!r} did not stack the {expected} mechanism")
    try:
        bounds = ov.omitted_variable_bounds(
            result, name, cf_y=CF_Y, cf_d=CF_D, rho=RHO, nu2_estimator="plugin"
        )
    except CapabilityError as refusal:
        # The corrected variance of a bound end is not positive.  CapabilityError stops a run,
        # so the study states that one refusal as the recordable error it is for one replicate.
        # Any other CapabilityError is a defect of the study and stops the run.
        key = _refusal_key_of(refusal)
        if key is None:
            raise
        raise DataError(f"the plug-in bound of {name!r} refused: {key}") from refusal
    if bounds.ci_lower is None or bounds.ci_upper is None:
        raise RuntimeError(f"the plug-in bound of {name!r} reported no limit")
    rebuilt = rebuild(result, scenario, token, elements)
    strength_value = bounds.confounding_strength
    psi = np.asarray(elements.psi_nu2, dtype=float)
    reported = {
        "nu2": _nu2_standard_error(result, elements, psi),
        "lower": (bounds.lower - bounds.ci_lower) / ONE_SIDED,
        "upper": (bounds.ci_upper - bounds.upper) / ONE_SIDED,
    }
    positive = _bound_variances(result, name, elements, rebuilt.curve, strength_value)
    own = {row: positive.standard_error(row, name) for row in CLAIM_ROWS}
    own["nu2"] = _nu2_standard_error(result, elements, rebuilt.curve)
    for row in ("nu2", "lower", "upper"):
        if abs(own[row] - reported[row]) > REBUILD_TOLERANCE * max(1.0, own[row]):
            raise RuntimeError(f"the rebuilt {row} standard error of {name!r} is not reported")
    estimates = {"lower": bounds.lower, "upper": bounds.upper}
    control: dict[str, float] | None = None
    if token in CONTROLLED:
        if scenario in ESTIMATED:
            changed = psi - rebuilt.mechanism_term
        else:
            changed = psi + study_mechanism_term(result, scenario, token)
        control = _control_standard_errors(
            positive, _bound_variances(result, name, elements, changed, strength_value)
        )
    cluster = result.data.cluster
    return ParameterValues(
        bounds={
            row: Bound(
                estimate=float(estimates[row]),
                std_error=float(reported[row]),
                control=None if control is None else float(control[row]),
                std_error_uncorrected=positive.uncorrected_standard_error(row),
                correction_share=positive.share(row),
            )
            for row in CLAIM_ROWS
        },
        nu2=Nu2(
            estimate=float(elements.nu2),
            std_error=float(reported["nu2"]),
            raw_estimate=float(elements.nu2) + float(elements.nu2_correction),
            raw_std_error=math.sqrt(float(influence_variance(psi, cluster))),
            correction=float(elements.nu2_correction),
            variance_correction=float(elements.nu2_variance_correction),
        ),
    )


#: The extra columns of a primary row that carry the unjudged ``nu^2`` reading.
NU2_COLUMNS = (
    "requested_replicates",
    "nu2_truth",
    "nu2_estimate",
    "nu2_std_error",
    "nu2_raw_estimate",
    "nu2_raw_std_error",
    "nu2_correction",
    "nu2_variance_correction",
)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    primary: str,
    replicate: int,
    requested: int = PRIMARY_REPLICATES,
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema, one row per bound end.

    Each row also carries the draw's unjudged ``nu^2`` reading in :data:`NU2_COLUMNS`;
    :func:`extra_artifacts` reads them, and the replicate file drops them.
    """
    scenario, token = PRIMARY[primary]
    result = fit(frame, scenario)
    values = parameter_rows(result, scenario, token)
    nu2 = values.nu2
    rows: list[dict[str, Any]] = []
    for row in CLAIM_ROWS:
        bound = values.bounds[row]
        key = estimand_key(token, row)
        reference = float(truth[key])
        low = bound.estimate - TWO_SIDED * bound.std_error
        high = bound.estimate + TWO_SIDED * bound.std_error
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": primary,
                "replicate": replicate,
                "n": result.n,
                "estimand": key,
                "truth": reference,
                "estimate": bound.estimate,
                "inference_estimate": bound.estimate,
                "std_error": bound.std_error,
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
                "requested_replicates": requested,
                "nu2_truth": float(truth[estimand_key(token, "nu2")]),
                "nu2_estimate": nu2.estimate,
                "nu2_std_error": nu2.std_error,
                "nu2_raw_estimate": nu2.raw_estimate,
                "nu2_raw_std_error": nu2.raw_std_error,
                "nu2_correction": nu2.correction,
                "nu2_variance_correction": nu2.variance_correction,
            }
        )
    return rows


# ---------------------------------------------------------------- the unjudged nu^2 reading

#: The columns of :data:`NU2_ROWS`.
NU2_ROW_COLUMNS = (
    "scenario",
    "parameter",
    "replicate",
    "n",
    "truth",
    "estimate",
    "std_error",
    "raw_estimate",
    "raw_std_error",
    "correction",
    "variance_correction",
)

#: The columns of :data:`NU2_READING`.
NU2_READING_COLUMNS = (
    "scenario",
    "parameter",
    "requested",
    "replicates",
    "refused",
    "coverage",
    "coverage_lower99",
    "coverage_upper99",
    "bias_sd",
    "se_ratio",
    "raw_coverage",
    "raw_bias_sd",
    "raw_se_ratio",
    "mean_correction",
)


def _wilson(covered: int, total: int, level: float = 0.99) -> tuple[float, float]:
    """The Wilson score interval of a proportion."""
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = covered / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def _read(truth: float, estimate: np.ndarray, std_error: np.ndarray) -> tuple[float, float, float]:
    """Coverage, bias in standard deviations, and SE ratio of one set of draws."""
    covered = int(np.sum(np.abs(estimate - truth) <= TWO_SIDED * std_error))
    spread = float(estimate.std(ddof=1))
    return (
        covered / len(estimate),
        (float(estimate.mean()) - truth) / spread,
        (float(std_error.mean()) / spread),
    )


def nu2_reading(rows: pd.DataFrame, requested: Mapping[tuple[str, str], int]) -> pd.DataFrame:
    """The measured reading of the corrected ``nu^2``, one row per scenario and parameter.

    ``coverage`` is the share of draws whose corrected interval holds the truth, with its 99%
    Wilson interval.  ``bias_sd`` is the mean estimate minus the truth over the standard
    deviation of the estimate, and ``se_ratio`` is the mean reported standard error over that
    deviation.  The ``raw_`` columns read the uncorrected :math:`E_n[\\hat\\alpha^2]` on the
    same draws.  Every column reads the draws the package did not refuse, and ``refused`` is
    ``requested`` less those.  Nothing here is a verdict.

    Parameters
    ----------
    rows : pandas.DataFrame
        The :data:`NU2_ROWS` frame.
    requested : Mapping
        The requested replicates of each ``(scenario, parameter)``.

    Returns
    -------
    pandas.DataFrame
        The reading, in :data:`NU2_READING_COLUMNS`.
    """
    out = []
    for (scenario, token), count in requested.items():
        cell = rows.loc[(rows["scenario"] == scenario) & (rows["parameter"] == token)]
        if cell.empty:
            raise ValueError(f"no surviving draw for {scenario}/{token}")
        truth = float(cell["truth"].iloc[0])
        estimate, error = cell["estimate"].to_numpy(), cell["std_error"].to_numpy()
        coverage, bias, ratio = _read(truth, estimate, error)
        raw_coverage, raw_bias, raw_ratio = _read(
            truth, cell["raw_estimate"].to_numpy(), cell["raw_std_error"].to_numpy()
        )
        low, high = _wilson(round(coverage * len(cell)), len(cell))
        out.append(
            {
                "scenario": scenario,
                "parameter": token,
                "requested": count,
                "replicates": len(cell),
                "refused": count - len(cell),
                "coverage": coverage,
                "coverage_lower99": low,
                "coverage_upper99": high,
                "bias_sd": bias,
                "se_ratio": ratio,
                "raw_coverage": raw_coverage,
                "raw_bias_sd": raw_bias,
                "raw_se_ratio": raw_ratio,
                "mean_correction": float(cell["correction"].mean()),
            }
        )
    return pd.DataFrame(out, columns=list(NU2_READING_COLUMNS))


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The unjudged ``nu^2`` reading, from the extra columns of the primary rows.

    :data:`NU2_ROWS` holds one row for each surviving draw, and :data:`NU2_READING` one row for
    each scenario and parameter.  The replicate file drops these columns, so the reading is the
    only place the corrected ``nu^2`` is published.  No verdict reads it.
    """
    lower = rows.loc[rows["estimand"].str.endswith("_lower")]
    scenarios = [scenario for scenario in PARAMETERS if scenario in set(_scenario_column(lower))]
    frame = pd.DataFrame(
        {
            "scenario": _scenario_column(lower),
            "parameter": [PRIMARY[name][1] for name in lower["scenario"]],
            "replicate": lower["replicate"].to_numpy(),
            "n": lower["n"].to_numpy(),
            "truth": lower["nu2_truth"].to_numpy(),
            "estimate": lower["nu2_estimate"].to_numpy(),
            "std_error": lower["nu2_std_error"].to_numpy(),
            "raw_estimate": lower["nu2_raw_estimate"].to_numpy(),
            "raw_std_error": lower["nu2_raw_std_error"].to_numpy(),
            "correction": lower["nu2_correction"].to_numpy(),
            "variance_correction": lower["nu2_variance_correction"].to_numpy(),
        }
    )
    order = {scenario: index for index, scenario in enumerate(PARAMETERS)}
    tokens = {
        scenario: {token: index for index, token in enumerate(PARAMETERS[scenario])}
        for scenario in PARAMETERS
    }
    frame = frame.assign(
        _s=frame["scenario"].map(order),
        _t=[tokens[s][t] for s, t in zip(frame["scenario"], frame["parameter"], strict=True)],
    )
    frame = frame.sort_values(["_s", "_t", "replicate"], ignore_index=True)
    frame = frame.loc[:, list(NU2_ROW_COLUMNS)]
    requested = {
        (PRIMARY[name][0], PRIMARY[name][1]): int(group["requested_replicates"].iloc[0])
        for name, group in lower.groupby("scenario", sort=False)
    }
    ordered = {
        (scenario, token): requested[(scenario, token)]
        for scenario in scenarios
        for token in PARAMETERS[scenario]
        if (scenario, token) in requested
    }
    return {NU2_ROWS: frame, NU2_READING: nu2_reading(frame, ordered)}


def _scenario_column(frame: pd.DataFrame) -> list[str]:
    return [PRIMARY[name][0] for name in frame["scenario"]]


def account_failures(failures: pd.DataFrame, artifacts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Check that the unjudged reading accounts for every fit the driver recorded as lost.

    The reading is a measurement and gates no claim, so this audit judges bookkeeping only.  For
    each scenario and parameter, ``refused`` in :data:`NU2_READING` must equal the replicates
    that the failure records lost.  A cell that disagrees is a failing row, and so is a lost
    cell with no reading row.
    """
    reading = artifacts[NU2_READING]
    lost = evidence_failures.expand(failures, STUDY)
    counts = lost.drop_duplicates(["scenario", "replicate"]).groupby("scenario").size().to_dict()
    refused = {
        primary_name(str(row.scenario), str(row.parameter)): int(row.refused)
        for row in reading.itertuples(index=False)
    }
    wrong = [
        {"scenario": name, "lost": counts.get(name, 0), "refused": refused.get(name, 0)}
        for name in sorted(set(counts) | set(refused))
        if counts.get(name, 0) != refused.get(name, 0)
    ]
    return evidence_failures.audits(
        {"nu2 reading refused counts equal the lost fits": pd.DataFrame(wrong)},
        missing={},
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    primary, index, n = key
    return draw_scenario(primary, n, index)


@dataclass(frozen=True)
class _Replicate:
    """The fit of one primary key, closed over the run's requested replicate count."""

    requested: int

    def __call__(
        self, key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
    ) -> list[dict[str, Any]]:
        primary, index, _ = key
        frame, truth = drawn
        return cleverly_rows(frame, truth, primary, index, self.requested)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication.

    A key is one parameter of one draw.  A refusal of the package fails that key alone.  The
    rows keep the extra columns that :func:`extra_artifacts` reads, and the driver drops them
    from the replicate file after the hook has run.
    """
    outcomes = replication.fit_primary(
        [(primary, index, n) for primary in STUDY.scenarios for index in range(replicates)],
        _Replicate(replicates),
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, [*REPLICATE_COLUMNS, *NU2_COLUMNS]])
