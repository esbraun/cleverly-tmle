"""Registered evidence for the cross-fitted stacked per-arm C-TMLE curve (X58 part c, X58b1).

The subject is ``CTMLE(strategy="oat", oat_design="per_arm", cross_fit=True, n_folds=5)`` with
:class:`~cleverly.learners.ParametricOutcome` and :class:`~cleverly.learners.ParametricMechanism`,
at ``repeats`` of 1 and 3, on a binary outcome.  Such a fit is an exact V-fold stacked
Z-estimator: one outcome and one mechanism fit per fold, a pooled fluctuation per arm and the
whole-sample mean per arm solve one system.  The fit reports the curve ``-A_n^{-1} U_i`` of that
system (:mod:`cleverly.estimators._oat_stack`; plan 13.3, decision D1), and an admitted estimate
carries the status ``"influence_curve"``.  Under ``repeats=3`` the reported estimate is the
median over draws with the split-adjusted variance of
:func:`~cleverly.inference.median_estimates`.  The study ``ctmle-oat-per-arm-parametric``
measures the in-sample form of the same curve (X58a).

**No comparator.**  ``drtmle``'s ``adapt_g`` reports the shipped curve, so no R program measures
the corrected uncertainty.  The study has ``reference=None``, runs no R, and its evidence page
labels the interval "Python property evidence, no comparator".

**The laws** are the laws of plan 13.6: ``binary_active`` of :mod:`tests.studies.oat_per_arm_laws`
(whose truth is a quadrature), its sharp-null twin, and ``binary_active`` with ``W1`` dropped from
the outcome features, whose limit ``psi*`` comes from :mod:`tests.studies.oat_parametric_quadrature`.
The generated-design pair pins ``Qbar_0`` of ``binary_active`` for its oracle arm.

**The primary phase** fits ``binary_active`` cross-fitted at ``n = 1,500`` for 800 replicates and
reports ``ey0``, ``ey1``, ``ate``, ``rr`` and ``or``.  The record needs a primary scenario.  Plan
13.6 counts no primary phase among the units of ``G`` for this study, because the in-sample
primary rows are X58a's.  Every other cell is a property cell of
:mod:`tests.studies.ctmle_oat_per_arm_parametric_crossfit_properties`.

**The declared reading** (plan 13.6, 5.4 carried over), fixed before any run:

* A red calibration row is a finite-sample result only under the AGENT-BRIEF rule: its 99%
  SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no
  defect.  Any other red row is a defect suspect, and it is diagnosed before anything is
  published.
* ``shipped_curve_control`` (``crossfit_1500``) and ``shipped_curve_median_control``
  (``repeats``) are negative controls.  Each *must* be red.  The first reads the SE ratio of the
  shipped adaptive-propensity curve of the same cross-fitted fits, expected at 0.845.  The second
  reads the median over draws of the stored shipped standard error with the split-adjusted
  variance.  In harness terms a control row publishes ``passed = True`` exactly when its SE-ratio
  interval leaves the calibration band on the declared side.  A control whose interval stays
  inside the band, or straddles it, publishes ``passed = False``.  Its cell then cannot see its
  term, the study is invalid, and it is redesigned before any reading.  The policy is ``gated``,
  so such a row blocks publication.  The study has no inflated control, so it takes no
  ``se_low`` clause binding (plan 13.6).
* The coverage of ``psi_0`` in the pseudo-true group, the ``sqrt(n)`` SD of the groups at
  ``n = 500`` and ``n = 8,000`` against the quadrature SD, the per-draw spread of the repeated
  fits and the paired deficit of the generated-design pair are reported and gate nothing.

**Failures.**  The record declares ``failure_bound={IMPLEMENTATION: 0.01}`` and the property
module declares ``FAILURE_BOUND = 0.01``, in the form ``docs/development/method-benchmarking.md``
("Property fitting failures", "The bound") defines.  A failed replicate is recorded and not
redrawn, and the run continues: the primary phase records it through ``fit_primary`` and the
property phase through ``fit_properties``.  A fit that raises, and an admitted fit whose stack
fails (``fit_cleverly`` raises on any status other than ``"influence_curve"``), both count as
failed replicates.  Under ``repeats=3`` the median carries the precedent parameter status of its
draws, so one failed draw fails the replicate.  A cell may lose ``floor(0.01 R)`` replicates of its ``R``, each clause is judged by its worst case
over the lost values, and one more lost replicate reads ``bound exceeded``.

**Sizes.**  Plan 13.6 sizes the study by the study-wise rule of section 12.  ``G`` counts nine
units: six calibration groups (the generated-design pair among them), one type I group, no primary
phase and two negative controls.  Each unit is sized so that its pass probability is at least
``0.95 ** (1 / 9) = 0.99432``, so the study passes with probability at least 0.95 when nothing is
wrong.  The budgets of :data:`BUDGETS` are the rev 3.1 expected budgets of plan 13.6, rounded up
to a multiple of 400.  The declared-size pilot (``reviews/x58/x58b/pilot`` of the plans
repository) reads every group at those budgets on pilot-only seeds and reports the study-wise
pass probability before any declared run.  A budget moves only by a later commit that cites that
pilot.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import CTMLE
from cleverly.exceptions import DataError
from cleverly.learners import ParametricMechanism, ParametricOutcome
from tests.parallel import STUDY_JOBS
from tests.studies import oat_parametric_quadrature as quad
from tests.studies import oat_per_arm_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import primary_rows

PRIMARY_REPLICATES = 800
PRIMARY_N = 1_500
#: Fresh entropy: no registered study uses either number, and
#: ``tests/unit/test_ctmle_oat_per_arm_parametric_crossfit_study_design.py`` checks that.
SEED = 20261168
RESAMPLING_SEED = 2026116801
IMPLEMENTATION = "cleverly-ctmle-oat-per-arm-parametric-crossfit"
SCENARIO = "binary_active"
#: The share of a cell's replicates that a run may lose, declared before the first run.  The
#: record carries it as ``failure_bound={IMPLEMENTATION: FAILURE_BOUND}`` and the property module
#: declares the same value.
FAILURE_BOUND = 0.01
#: The factor of ``shrunken_se_control``.  The shared ``calibration_controls`` value.
SHRUNKEN_SE_FACTOR = 0.70
#: The fold count the support predicate studies before RM42b, and the repeat count of the
#: ``repeats`` group.
FOLDS = 5
REPEATS = 3

BINARY_ESTIMANDS = ("ey0", "ey1", "ate", "rr", "or")
ARM_ESTIMANDS = ("ey0", "ey1", "ate")

# ---------------------------------------------------------------------------------- the sizes

#: The replicates of each fit group: the rev 3.1 expected budgets of plan 13.6, each a multiple
#: of 400.  The six calibration groups are sized to the per-unit target ``0.95 ** (1 / 9)``.
#: ``crossfit_null`` is sized at 1,200: at most 93 rejections pass the type I verdict, and the
#: pass probability is 0.99998 at a true rate of 0.05 and 0.99953 at 0.055.
BUDGETS: Mapping[str, int] = {
    "crossfit_1500": 15_200,
    "crossfit_500": 64_800,
    "crossfit_8000": 10_000,
    "crossfit_pseudo_true": 10_000,
    "crossfit_generated_design": 10_800,
    "repeats": 10_800,
    "crossfit_null": 1_200,
}
#: The units of the study-wise rule (plan 13.6): calibration groups, type I groups, the primary
#: phase and the declared negative controls.
UNITS: Mapping[str, int] = {
    "calibration_groups": 6,
    "type_i_groups": 1,
    "primary_phase": 0,
    "negative_controls": 2,
}
UNIT_COUNT = sum(UNITS.values())
NULL_N = 1_000
CALIBRATION_N = 1_500


@dataclass(frozen=True)
class Fit:
    """One cross-fitted ``CTMLE`` configuration of the study.

    Parameters
    ----------
    name : str
        The configuration's name inside its group.
    estimands : tuple of str
        The estimands the fit requests.
    outcome : {"parametric", "oracle"}
        ``"parametric"`` fits :class:`~cleverly.learners.ParametricOutcome`, which the support
        predicate admits.  ``"oracle"`` pins ``Qbar_0`` of ``binary_active``, which the predicate
        refuses, so the fit keeps the shipped curve and its plug-in status.
    repeats : int
        The number of cross-fitting draws.
    covariates : tuple of str
        The covariate columns.
    """

    name: str
    estimands: tuple[str, ...] = ARM_ESTIMANDS
    outcome: str = "parametric"
    repeats: int = 1
    covariates: tuple[str, ...] = ("W1", "W2", "W3")


@dataclass(frozen=True)
class Group:
    """The cells that share one sample per replicate and the fits made on it.

    Parameters
    ----------
    name : str
        The group's name; it labels the checkpoints and the sample stream.
    law : str
        The law the sample is drawn from (:func:`sample`).
    n : int
        The sample size.
    fits : tuple of Fit
        The configurations fitted on each sample.
    """

    name: str
    law: str
    n: int
    fits: tuple[Fit, ...]

    @property
    def replicates(self) -> int:
        """The group's budget, :data:`BUDGETS`."""
        return BUDGETS[self.name]


MAIN = Fit("main")
RATIOS = Fit("main", estimands=(*ARM_ESTIMANDS, "rr", "or"))

GROUPS: Mapping[str, Group] = {
    group.name: group
    for group in (
        Group("crossfit_1500", "binary_active", CALIBRATION_N, (RATIOS,)),
        Group("crossfit_500", "binary_active", 500, (MAIN,)),
        Group("crossfit_8000", "binary_active", 8_000, (MAIN,)),
        Group(
            "crossfit_pseudo_true",
            "binary_active",
            CALIBRATION_N,
            (Fit("main", covariates=("W2", "W3")),),
        ),
        Group(
            "crossfit_generated_design",
            "binary_active",
            CALIBRATION_N,
            (
                Fit("estimated", estimands=("ate",)),
                Fit("oracle", estimands=("ate",), outcome="oracle"),
            ),
        ),
        Group("repeats", "binary_active", CALIBRATION_N, (Fit("main", repeats=REPEATS),)),
        Group("crossfit_null", "binary_null", NULL_N, (Fit("main", estimands=("ate",)),)),
    )
}


# ---------------------------------------------------------------------------------- the cells


@dataclass(frozen=True)
class Cell:
    """One published property cell.

    Parameters
    ----------
    plan : str
        The group of plan 13.6 that this cell belongs to.
    family : str
        The property family the harness classifies the cell under.
    name : str
        The published cell name.
    group : str
        The fit group whose sample and fit the cell reads.
    fit : str
        The configuration of the group the cell reads.
    estimand : {"ey0", "ey1", "ate", "log_rr", "log_or"}
        The estimand; a ratio is read on the log scale.
    truth : str
        The law whose arm means define the cell's truth (:func:`truth`).
    recipe : {"corrected", "shipped_curve"}
        How the row's standard error is built.  ``"corrected"`` reads the estimate.
        ``"shipped_curve"`` reads the shipped adaptive-propensity curve of the same fit, as the
        median over draws with the split-adjusted variance.
    role : {"positive", "control"}
        The row's role.
    report : str
        A quantity the row also records and no verdict reads.  ``"psi0"``: whether the interval
        covers ``psi_0``.  ``"root_n"``: the ``sqrt(n)`` SD against the quadrature SD, read at
        summary.  ``"spread"``: the SD of the draws' point estimates.
    """

    plan: str
    family: str
    name: str
    group: str
    fit: str
    estimand: str
    truth: str
    recipe: str = "corrected"
    role: str = "positive"
    report: str = ""


def _calibration(
    label: str,
    truth_law: str,
    *,
    group: str | None = None,
    estimands: tuple[str, ...] = ("ey0", "ey1", "ate"),
    report: str = "",
) -> list[Cell]:
    """The positive ``<label>__<estimand>__corrected`` cells of one group's fit."""
    return [
        Cell(
            label,
            "interval_calibration",
            f"{label}__{estimand}__corrected",
            group or label,
            "main",
            estimand,
            truth_law,
            report=report,
        )
        for estimand in estimands
    ]


def _cells() -> tuple[Cell, ...]:
    out: list[Cell] = []
    out += _calibration(
        "crossfit_1500",
        "binary_active",
        estimands=("ey0", "ey1", "ate", "log_rr", "log_or"),
    )
    out.append(
        Cell(
            "crossfit_1500",
            "interval_calibration",
            "crossfit_1500__ate__shipped_curve__shrunken_se_control",
            "crossfit_1500",
            "main",
            "ate",
            "binary_active",
            recipe="shipped_curve",
            role="control",
        )
    )
    out += _calibration("crossfit_500", "binary_active", report="root_n")
    out += _calibration("crossfit_8000", "binary_active", report="root_n")
    out += _calibration("crossfit_pseudo_true", "pseudo_true", report="psi0")
    out += [
        Cell(
            "crossfit_generated_design",
            "generated_design",
            f"{GENERATED_LABEL}__{name}",
            "crossfit_generated_design",
            fit,
            "ate",
            "binary_active",
        )
        for name, fit in (("oracle_design", "oracle"), ("estimated", "estimated"))
    ]
    out += _calibration("repeats", "binary_active", report="spread")
    out.append(
        Cell(
            "repeats",
            "interval_calibration",
            "repeats__ate__shipped_curve__shrunken_se_control",
            "repeats",
            "main",
            "ate",
            "binary_active",
            recipe="shipped_curve",
            role="control",
        )
    )
    out.append(
        Cell(
            "crossfit_null",
            "type_i_error",
            "sharp_null",
            "crossfit_null",
            "main",
            "ate",
            "binary_null",
        )
    )
    return tuple(out)


#: The label of the generated-design pair's cells: ``binary_active__oracle_design`` and
#: ``binary_active__estimated``, both cross-fitted.
GENERATED_LABEL = "binary_active"

CELLS: tuple[Cell, ...] = _cells()

#: The calibration cells that the harness derives two controls from, without a fit: the
#: ``shrunken_se_control`` and the ``noise_control`` of each group's ``ate`` row.
CONTROL_LABELS: tuple[str, ...] = (
    "crossfit_1500__ate",
    "crossfit_500__ate",
    "crossfit_8000__ate",
    "crossfit_pseudo_true__ate",
    "repeats__ate",
)
DERIVED_CONTROLS = tuple(
    f"{label}__{kind}"
    for label in CONTROL_LABELS
    for kind in ("shrunken_se_control", "noise_control")
)

#: The groups of plan 13.6 for this study, each with the group it reads.  The design test pins
#: the list against the plan.
PLAN_CELLS: Mapping[str, str] = {
    "crossfit_1500": "crossfit_1500",
    "crossfit_500": "crossfit_500",
    "crossfit_8000": "crossfit_8000",
    "crossfit_pseudo_true": "crossfit_pseudo_true",
    "crossfit_generated_design": "crossfit_generated_design",
    "repeats": "repeats",
    "crossfit_null": "crossfit_null",
}

#: The negative controls that must be red (plan 13.6), by published cell.
NEGATIVE_CONTROLS: Mapping[str, str] = {
    "crossfit_1500__ate__shipped_curve__shrunken_se_control": "below",
    "repeats__ate__shipped_curve__shrunken_se_control": "below",
}


def property_cells() -> dict[str, tuple[str, ...]]:
    """The cells the committed property summary must contain."""
    out: dict[str, list[str]] = {}
    for cell in CELLS:
        out.setdefault(cell.family, []).append(cell.name)
    out["interval_calibration"].extend(DERIVED_CONTROLS)
    return {family: tuple(sorted(names)) for family, names in sorted(out.items())}


def cells_of(group: str) -> tuple[Cell, ...]:
    """The cells that read ``group``'s fits."""
    return tuple(cell for cell in CELLS if cell.group == group)


STUDY = StudyRecord(
    name="cross-fitted outcome-adaptive per-arm C-TMLE with a parametric outcome",
    slug="ctmle-oat-per-arm-parametric-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "ctmle_oat_per_arm_parametric_crossfit",
    document=(
        "docs/technical-reference/method-evidence/"
        "outcome-adaptive-per-arm-c-tmle-parametric-crossfit.md"
    ),
    anchor="outcome-adaptive-per-arm-c-tmle-parametric-crossfit",
    scenarios={SCENARIO: BINARY_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/studies/canonical_ctmle_oat_per_arm_parametric_crossfit.py",
        "tests/studies/ctmle_oat_per_arm_parametric_crossfit_properties.py",
        "tests/studies/oat_parametric_quadrature.py",
        "tests/studies/oat_per_arm_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_ctmle_oat_per_arm_parametric_crossfit",
    properties_module="tests.studies.ctmle_oat_per_arm_parametric_crossfit_properties",
    property_cells=property_cells(),
    publication_policy="gated",
    calibration_efficiency_ratio=False,
)

CONFIGURATION: dict[str, Any] = {
    "strategy": "oat",
    "oat_design": "per_arm",
    "cross_fit": True,
    "n_folds": FOLDS,
    "repeats_by_group": {"repeats": REPEATS},
    "simultaneous_intervals": False,
    "outcome_learner": (
        "ParametricOutcome: unpenalized logistic GLM on [A indicators, W, A x W], one fit per fold"
    ),
    "treatment_learner": (
        "ParametricMechanism: per arm and fold, unpenalized logistic regression of 1{A = a} on "
        "the cubic in l = logit(clip(Qbar_k(a, W), 1e-6, 1 - 1e-6))"
    ),
    "g_bounds": list(laws.G_BOUNDS),
    "covariates_by_group": {"crossfit_pseudo_true": ["W2", "W3"]},
    "max_iter": 100,
    "tol": 1e-10,
    "random_state": 0,
    "primary": {
        "scenario": SCENARIO,
        "n": PRIMARY_N,
        "replicates": PRIMARY_REPLICATES,
        "estimands": list(BINARY_ESTIMANDS),
        "law": "binary_active of tests.studies.oat_per_arm_laws",
    },
    "comparator": "none: Python property evidence, no comparator",
    "comparator_reason": (
        "drtmle's adapt_g reports the shipped curve, so no R program measures the corrected "
        "curve; the point estimate of the in-sample form agrees with the registered pipelines "
        "through test_parametric_learners.py"
    ),
    "budgets": dict(BUDGETS),
    "units": dict(UNITS),
    "budget_rule": (
        "plan 13.6 (rev 3.1) expected budgets, each a multiple of 400; the study-wise rule of "
        "plan section 12 with G = 9 gated units, each sized to pass with probability >= "
        "0.95 ** (1 / 9); crossfit_null is sized at 1,200 (at most 93 rejections pass)"
    ),
    "failure_bounds": {
        "declared": FAILURE_BOUND,
        "applied": (
            "0.01 in both phases: a cell may lose floor(0.01 R) replicates, and each clause "
            "is judged by its worst case over the lost values; one more lost replicate reads "
            "bound exceeded. A fit that raises, and an admitted fit whose stack status is not "
            "influence_curve, both count as failed replicates"
        ),
    },
    "shrunken_se_factor": SHRUNKEN_SE_FACTOR,
    "calibration_efficiency_ratio": (
        "False: the noise control reads the SE-ratio rule, and no cell claims a ratio "
        "against an efficiency bound"
    ),
    "declared_reading": (
        "A red calibration row is finite-sample only under the AGENT-BRIEF rule (its 99% "
        "SE-ratio interval contains 1, its coverage is consistent with 0.95, and diagnosis "
        "finds no defect). shipped_curve_control and shipped_curve_median_control must be "
        "red, that is, their SE-ratio intervals must leave the calibration band below. The "
        "harness publishes passed=True for such a control. If either control stays inside "
        "or straddles the band, its cell cannot see its term, the study is invalid, and it is "
        "redesigned before any reading. The coverage of psi_0 in the pseudo-true group, the "
        "sqrt(n) SD of the groups at n = 500 and n = 8,000, the per-draw spread of the "
        "repeated fits and the paired deficit of the generated-design pair are reported and "
        "gate nothing."
    ),
    "negative_controls": dict(NEGATIVE_CONTROLS),
    "negative_control_expected_se_ratio": {
        "shipped_curve_control": 0.845,
    },
    "publication_reading": (
        "gated: every row must pass, and a negative control passes by being red"
    ),
    "family_mapping": (
        "the groups are interval_calibration cells named <label>__<estimand>__corrected, so "
        "the shared calibration verdict applies per estimand"
    ),
}


# ---------------------------------------------------------------------------------- the laws


def sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of ``law`` (columns ``W1``, ``W2``, ``W3``, ``A``, ``Y``)."""
    if law == "binary_active":
        return laws.BINARY_ACTIVE.sample(n, seed)[0]
    if law == "binary_null":
        return laws.BINARY_NULL.sample(n, seed)[0]
    raise KeyError(law)


@cache
def arm_means(law: str) -> tuple[float, float]:
    """``(psi_0, psi_1)`` of ``law``: the true arm means, or the limit ``psi*`` for ``pseudo_true``.

    ``pseudo_true`` is ``binary_active`` fitted without ``W1`` in the outcome features, whose
    limit is the quadrature's ``psi*``.
    """
    if law in ("binary_active", "binary_null"):
        truth = (laws.BINARY_ACTIVE if law == "binary_active" else laws.BINARY_NULL).truth()
        return float(truth["ey0"]), float(truth["ey1"])
    if law == "pseudo_true":
        star = quad.population_fit(quad.BINARY_ACTIVE, covariates=(1, 2)).psi_star
        return float(star[0]), float(star[1])
    raise KeyError(law)


def truth(law: str, estimand: str) -> float:
    """The truth of ``estimand`` on ``law``; ``log_rr`` and ``log_or`` are on the log scale."""
    low, high = arm_means(law)
    if estimand == "ey0":
        return low
    if estimand == "ey1":
        return high
    if estimand == "ate":
        return high - low
    if estimand == "log_rr":
        return math.log(high / low)
    if estimand == "log_or":
        return math.log(high / (1.0 - high)) - math.log(low / (1.0 - low))
    raise KeyError(estimand)


@cache
def curve_sd() -> dict[str, float]:
    """The asymptotic ``sqrt(n)`` SD of the corrected curve of ``binary_active``, by estimand.

    The quadrature's ``ey0`` and ``ey1`` SDs and the ATE SD (1.1234).  They scale the noise
    control and the reported ``sqrt(n)`` SD of the large and small groups.  The cross-fitted
    curve has the same limit (plan 13.3, check CW4).
    """
    fit = quad.population_fit(quad.BINARY_ACTIVE)
    variance = quad.covariance(fit)
    return {
        "ey0": float(np.sqrt(variance[0, 0])),
        "ey1": float(np.sqrt(variance[1, 1])),
        "ate": quad.ate_sd(fit),
    }


def control_bounds() -> dict[str, float]:
    """The scale of each derived ``noise_control``: the ATE SD at the group's own ``n``.

    ``calibration_controls`` divides the bound by ``sqrt(CALIBRATION_N)``.  A group at another
    ``n`` rescales it by ``sqrt(CALIBRATION_N / n)``, so every ``noise_control`` adds one
    standard deviation of the group's own estimate.
    """
    sizes = {group.name: group.n for group in GROUPS.values()}
    sd = curve_sd()["ate"]
    return {
        label: sd * math.sqrt(CALIBRATION_N / sizes[label.removesuffix("__ate")])
        for label in CONTROL_LABELS
    }


# ---------------------------------------------------------------------------------- the fit


def estimator(fit: Fit) -> CTMLE:
    """The subject of one configuration: a cross-fitted per-arm ``CTMLE``, parametric learners."""
    outcome_learner = (
        ParametricOutcome() if fit.outcome == "parametric" else laws.OracleOutcome("binary_active")
    )
    return CTMLE(
        strategy="oat",
        oat_design="per_arm",
        outcome_learner=outcome_learner,
        treatment_learner=ParametricMechanism(degree=3),
        cross_fit=True,
        n_folds=FOLDS,
        repeats=fit.repeats,
        simultaneous=False,
        estimands=fit.estimands,
        g_bounds=laws.G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def fit_cleverly(frame: pd.DataFrame, fit: Fit) -> Any:
    """Fit the study's subject on one sample.

    An admitted fit must report ``"influence_curve"`` for every requested estimand.  A
    ``"stack_failure_plugin"`` estimate is a failure of the replicate: the study measures the
    stacked curve, and a fall back to the shipped curve would measure a different estimator.
    Under ``repeats=3`` the median reports the precedent parameter status of its draws, so one
    failed draw fails the replicate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            estimator(fit)
            .fit(frame, outcome="Y", treatment="A", covariates=list(fit.covariates))
            .single()
        )
    if fit.outcome == "parametric":
        for name in fit.estimands:
            status = result[name].inference
            if status != "influence_curve":
                raise DataError(
                    f"{name} reports the inference status {status!r}, not the stacked curve"
                )
    return result


def initials(result: Any) -> dict[str, float]:
    """The untargeted plug-in of each primary estimand."""
    arms = result.data.arm_codes
    low, high = (float(np.mean(result.nuisance.outcome.arms[code])) for code in arms)

    def odds(p: float) -> float:
        return p / (1.0 - p)

    return {
        "ey0": low,
        "ey1": high,
        "ate": high - low,
        "rr": high / low,
        "or": odds(high) / odds(low),
    }


# ------------------------------------------------------------------------- the primary phase


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample and its truth from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truths = laws.BINARY_ACTIVE.sample(n, seed)
    return frame, {name: float(truths[name]) for name in BINARY_ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


MAIN_FIT = Fit("primary", estimands=BINARY_ESTIMANDS)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truths = drawn
    result = fit_cleverly(frame, MAIN_FIT)
    return primary_rows(
        result=result,
        truth=truths,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=BINARY_ESTIMANDS,
        initials=initials(result),
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every primary replication."""
    outcomes = replication.fit_primary(
        [(SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])
