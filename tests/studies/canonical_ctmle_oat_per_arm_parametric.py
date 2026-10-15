"""Registered evidence for the stacked per-arm C-TMLE curve, ``ctmle-oat-per-arm-parametric``.

The subject is ``CTMLE(strategy="oat", oat_design="per_arm", cross_fit=False)`` with
:class:`~cleverly.learners.ParametricOutcome` and :class:`~cleverly.learners.ParametricMechanism`
(X58 part a).  Such a fit is a finite-dimensional stacked Z-estimator: the outcome GLM, each
arm's mechanism, each arm's fluctuation and each arm's average solve one system.  The fit reports
the corrected curve ``-A_n^{-1} U_i`` of :mod:`cleverly.estimators._oat_stack`, and an admitted
estimate carries the status ``"influence_curve"``.  The registered study ``ctmle-oat-per-arm``
fits sklearn learners, so its fits stay on the plug-in status and no number of it moves.

**No comparator.**  ``drtmle``'s ``adapt_g`` reports the shipped curve, so no R program measures
the corrected uncertainty.  The study has ``reference=None``, runs no R, and its evidence page
labels the interval "Python property evidence, no comparator".  The point estimate agrees with
the registered pipelines to ``4e-14`` (``tests/unit/test_parametric_learners.py``).

**The laws** are the laws of the plan, section 5.1: ``binary_active`` of
:mod:`tests.studies.oat_per_arm_laws` (the primary, whose truth is a quadrature), its sharp-null
twin, and five quadrature laws of :mod:`tests.studies.oat_parametric_quadrature` (shared
prognosis, U-shaped, steep, the bounded outcome ``Y = 2 + 8 B``, and ``binary_active`` with
``W1`` dropped from the outcome features).

**The primary phase** fits ``binary_active`` at ``n = 1,500`` for 800 replicates and reports
``ey0``, ``ey1``, ``ate``, ``rr`` and ``or``.  Every other cell is a property cell of
:mod:`tests.studies.ctmle_oat_per_arm_parametric_properties`, sized by the plan's section 11.

**The declared reading** (plan 5.4), fixed before any run:

* A red calibration row is a finite-sample result only under the AGENT-BRIEF rule: its 99%
  SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no
  defect.  Any other red row is a defect suspect, and it is diagnosed before anything is
  published.
* ``shipped_curve_control`` and ``cross_term_dropped_control`` are negative controls.  Each
  *must* be red: the SE ratio of the shipped curve is expected at 0.845 and the SE ratio of the
  ATE built without the cross-arm covariance at 1.228.  In harness terms a control row
  publishes ``passed = True`` exactly when its SE-ratio interval leaves the calibration band on
  the declared side.  A control whose interval stays inside the band, or straddles it, publishes
  ``passed = False``.  Its cell then cannot see its term, the study is invalid, and it is
  redesigned before any reading.  The policy is ``gated``, so such a row blocks publication.
* The coverage of ``psi_0`` in the pseudo-true cell (expected near 0.56 for ``ey0``, 0.58 for
  ``ey1`` and 0.31 for ``ate``, from the declared-size pilot on pilot-only seeds), the ``sqrt(n)`` SD of the ladder against the
  quadrature SD, the clipped share of the steep cell, and the Monte Carlo covariance of the two
  arm means against the mean reported covariance are reported and gate nothing.  The covariance
  enters the gate through the ATE row of the shared-prognosis cell, which the dropped cross term
  moves from 1.007 to 1.238.

**Failures.**  The record declares ``failure_bound={IMPLEMENTATION: 0.01}`` and the property
module declares ``FAILURE_BOUND = 0.01``, in the form ``docs/development/method-benchmarking.md``
("Property fitting failures", "The bound") defines.  A failed replicate is recorded and not
redrawn, and the run continues: the primary phase records it through ``fit_primary`` and the
property phase through ``fit_properties``.  A fit that raises, and an admitted fit whose stack
fails (``fit_cleverly`` raises on any status other than ``"influence_curve"``), both count as
failed replicates.  Dropping such a replicate would select on success, and substituting the
shipped standard error would mix two estimators.  A cell may lose ``floor(0.01 R)`` replicates
of its ``R``.  Each clause of a cell inside that allowance is judged by its worst case over the
lost values (``property-failure-accounting.csv``), and one more lost replicate reads ``bound
exceeded`` before any clause runs.  The run reports the status, the seed and the group of each
failure (``failures.csv`` and ``property-failures.csv``), and the failure is diagnosed.  A bound
is raised only while no committed run shows a failure.  The pilots (5,000 draws per cell, then
every group at its declared budget on pilot-only seeds) raised no separation refusal and no
stack failure.

**Sizes.**  Each calibration group is sized so that the whole study passes with probability at
least 0.95 when nothing is wrong: each of the 12 gated calibration groups gets the per-group
target ``0.95 ** (1 / 12)``.  The plan's section 11 sized each cell to 0.95 on its own, which
gives a study-wise pass probability near 0.63 (pre-queue design review).  The predictive model
centres each row on its own pilot ratio with variance ``s^2 + tau_80^2``, ``tau_80 = 0.0080``
pooled over the study's gated rows.  Centring on the row, not on the arm-pooled mean of the
project rule, is the more conservative choice for a low row.
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
#: ``tests/unit/test_ctmle_oat_per_arm_parametric_study_design.py`` checks that.
SEED = 20261158
RESAMPLING_SEED = 2026115801
IMPLEMENTATION = "cleverly-ctmle-oat-per-arm-parametric"
SCENARIO = "binary_active"
#: The share of a cell's replicates that a run may lose, declared before the first run.  The
#: record carries it as ``failure_bound={IMPLEMENTATION: FAILURE_BOUND}`` and the property module
#: declares the same value.
FAILURE_BOUND = 0.01
#: The factor of ``shrunken_se_control``.  The shared ``calibration_controls`` value.
SHRUNKEN_SE_FACTOR = 0.70

BINARY_ESTIMANDS = ("ey0", "ey1", "ate", "rr", "or")
ARM_ESTIMANDS = ("ey0", "ey1", "ate")

# ---------------------------------------------------------------------------------- the sizes

#: The replicates of each fit group.  The 12 gated calibration groups, ``generated_design``
#: among them, are sized by the study-wise rule of the module docstring ("Sizes") from the
#: pilots, and each was read at its budget on pilot-only seeds before this declaration.  Two
#: groups are sized below.
#:
#: * ``sharp_null``: 800, the registered null size at ``n = 1,000``.
#: * ``power``: 1,200 at ``n = 800``.  The exact-law power there is 0.8475, and the chance that
#:   the lower Clopper-Pearson endpoint of the rejection rate clears 0.80 is 0.957 at 1,200
#:   replicates (0.800 at 800, 0.991 at 1,600).
BUDGETS: Mapping[str, int] = {
    "binary_active": 17_200,
    "ladder_500": 63_600,
    "ladder_2000": 8_400,
    "ladder_8000": 9_200,
    "shared_prognosis": 10_000,
    "kappa_law": 19_600,
    "bounds_active": 12_400,
    "auto_bounds": 9_200,
    "bounded_outcome": 16_400,
    "pseudo_true": 7_600,
    "generated_design": 9_200,
    "sharp_null": 800,
    "power": 1_200,
}
POWER_N = 800
NULL_N = 1_000
CALIBRATION_N = 1_500
#: The steep law's declared ``g_bounds``: it clips 12.4% of each arm's rows at the limit.
STEEP_BOUNDS = (0.05, 0.95)
BOUNDED_OUTCOME = (2.0, 10.0)


@dataclass(frozen=True)
class Fit:
    """One ``CTMLE`` configuration of the study.

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
    degree : int
        The degree of the mechanism basis in ``logit q``.
    g_bounds : tuple of float or str
        The declared ``g_bounds``, or ``"auto"``.
    q_bounds : tuple of float or None
        The declared outcome bounds of a bounded outcome.
    covariates : tuple of str
        The covariate columns.
    """

    name: str
    estimands: tuple[str, ...] = ARM_ESTIMANDS
    outcome: str = "parametric"
    degree: int = 3
    g_bounds: tuple[float, float] | str = laws.G_BOUNDS
    q_bounds: tuple[float, float] | None = None
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
        Group("binary_active", "binary_active", CALIBRATION_N, (MAIN,)),
        Group("ladder_500", "binary_active", 500, (MAIN,)),
        Group("ladder_2000", "binary_active", 2_000, (MAIN,)),
        Group("ladder_8000", "binary_active", 8_000, (MAIN,)),
        Group("shared_prognosis", "shared_prognosis", CALIBRATION_N, (RATIOS,)),
        Group(
            "kappa_law",
            "u_shaped",
            CALIBRATION_N,
            (Fit("cubic"), Fit("linear", degree=1)),
        ),
        Group(
            "bounds_active",
            "steep",
            CALIBRATION_N,
            (Fit("main", g_bounds=STEEP_BOUNDS),),
        ),
        Group("auto_bounds", "binary_active", CALIBRATION_N, (Fit("main", g_bounds="auto"),)),
        Group(
            "bounded_outcome",
            "bounded",
            CALIBRATION_N,
            (Fit("main", q_bounds=BOUNDED_OUTCOME),),
        ),
        Group(
            "pseudo_true",
            "binary_active",
            CALIBRATION_N,
            (Fit("main", covariates=("W2", "W3")),),
        ),
        Group(
            "generated_design",
            "binary_active",
            CALIBRATION_N,
            (
                Fit("estimated", estimands=("ate",)),
                Fit("oracle", estimands=("ate",), outcome="oracle"),
            ),
        ),
        Group("sharp_null", "binary_null", NULL_N, (Fit("main", estimands=("ate",)),)),
        Group("power", "binary_active", POWER_N, (Fit("main", estimands=("ate",)),)),
    )
}


# ---------------------------------------------------------------------------------- the cells


@dataclass(frozen=True)
class Cell:
    """One published property cell.

    Parameters
    ----------
    plan : str
        The cell of the plan, section 5.2, that this cell belongs to.
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
    recipe : {"corrected", "shipped_curve", "cross_term_dropped"}
        How the row's standard error is built.  ``"corrected"`` reads the estimate.
        ``"shipped_curve"`` reads the shipped adaptive-propensity curve of the same fit.
        ``"cross_term_dropped"`` reads ``sqrt(se_0^2 + se_1^2)``.
    role : {"positive", "control"}
        The row's role.
    report : str
        A quantity the row also records and no verdict reads.  ``"psi0"``: whether the interval
        covers ``psi_0``.  ``"covariance"``: both arm estimates and the reported covariance.
        ``"clipped"``: the share of rows at a declared ``g_bounds`` limit.
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
    plan: str,
    label: str,
    group: str,
    truth_law: str,
    *,
    fit: str = "main",
    estimands: tuple[str, ...] = ("ey0", "ey1", "ate"),
    report: str = "",
) -> list[Cell]:
    """The positive ``<label>__<estimand>__corrected`` cells of one fit."""
    return [
        Cell(
            plan,
            "interval_calibration",
            f"{label}__{estimand}__corrected",
            group,
            fit,
            estimand,
            truth_law,
            report=report if report != "covariance" or estimand == "ate" else "",
        )
        for estimand in estimands
    ]


def _cells() -> tuple[Cell, ...]:
    out: list[Cell] = []
    out += _calibration(
        "binary_active__corrected", "binary_active", "binary_active", "binary_active"
    )
    out.append(
        Cell(
            "shipped_curve_control",
            "interval_calibration",
            "binary_active__ate__shipped_curve__shrunken_se_control",
            "binary_active",
            "main",
            "ate",
            "binary_active",
            recipe="shipped_curve",
            role="control",
        )
    )
    for size in (500, 2_000, 8_000):
        out += _calibration(f"n_{size}", f"n_{size}", f"ladder_{size}", "binary_active")
    shared = _calibration(
        "shared_prognosis__corrected",
        "shared_prognosis",
        "shared_prognosis",
        "shared_prognosis",
        estimands=("ate",),
        report="covariance",
    )
    out += shared
    out.append(
        Cell(
            "cross_term_dropped_control",
            "interval_calibration",
            "shared_prognosis__ate__cross_term_dropped__inflated_se_control",
            "shared_prognosis",
            "main",
            "ate",
            "shared_prognosis",
            recipe="cross_term_dropped",
            role="control",
        )
    )
    out += _calibration(
        "shared_prognosis__ratios",
        "shared_prognosis",
        "shared_prognosis",
        "shared_prognosis",
        estimands=("log_rr", "log_or"),
    )
    for basis in ("cubic", "linear"):
        out += _calibration(
            "kappa_law__corrected",
            f"kappa_{basis}",
            "kappa_law",
            "u_shaped",
            fit=basis,
        )
    out += _calibration(
        "bounds_active__corrected", "bounds_active", "bounds_active", "steep", report="clipped"
    )
    out += _calibration("auto_bounds__corrected", "auto_bounds", "auto_bounds", "binary_active")
    out += _calibration(
        "bounded_outcome__corrected", "bounded_outcome", "bounded_outcome", "bounded"
    )
    out += _calibration(
        "pseudo_true__corrected", "pseudo_true", "pseudo_true", "pseudo_true", report="psi0"
    )
    out += [
        Cell(
            "oracle_design_against_estimated",
            "generated_design",
            f"binary_active__{name}",
            "generated_design",
            fit,
            "ate",
            "binary_active",
        )
        for name, fit in (("oracle_design", "oracle"), ("estimated", "estimated"))
    ]
    out += [
        Cell(
            "sharp_null", "type_i_error", "sharp_null", "sharp_null", "main", "ate", "binary_null"
        ),
        Cell("alternative", "power", "alternative", "power", "main", "ate", "binary_active"),
    ]
    return tuple(out)


CELLS: tuple[Cell, ...] = _cells()

#: The calibration cells that the harness derives two controls from, without a fit: the
#: ``shrunken_se_control`` and the ``noise_control`` of the primary ``ate`` row.
CONTROL_LABELS: tuple[str, ...] = ("binary_active__ate",)
DERIVED_CONTROLS = tuple(
    f"{label}__{kind}"
    for label in CONTROL_LABELS
    for kind in ("shrunken_se_control", "noise_control")
)

#: The plan's section 5.2 cells, in order, each with the group it reads.  The design test pins
#: the list against the plan.
PLAN_CELLS: Mapping[str, str] = {
    "binary_active__corrected": "binary_active",
    "shipped_curve_control": "binary_active",
    "n_500": "ladder_500",
    "n_2000": "ladder_2000",
    "n_8000": "ladder_8000",
    "shared_prognosis__corrected": "shared_prognosis",
    "cross_term_dropped_control": "shared_prognosis",
    "shared_prognosis__ratios": "shared_prognosis",
    "kappa_law__corrected": "kappa_law",
    "bounds_active__corrected": "bounds_active",
    "auto_bounds__corrected": "auto_bounds",
    "bounded_outcome__corrected": "bounded_outcome",
    "pseudo_true__corrected": "pseudo_true",
    "oracle_design_against_estimated": "generated_design",
    "sharp_null": "sharp_null",
    "alternative": "power",
}

#: The negative controls that must be red (plan 5.4), by published cell.
NEGATIVE_CONTROLS: Mapping[str, str] = {
    "binary_active__ate__shipped_curve__shrunken_se_control": "below",
    "shared_prognosis__ate__cross_term_dropped__inflated_se_control": "above",
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
    name="outcome-adaptive per-arm C-TMLE with a parametric outcome",
    slug="ctmle-oat-per-arm-parametric",
    artifacts=ROOT / "tests" / "canonical" / "ctmle_oat_per_arm_parametric",
    document=(
        "docs/technical-reference/method-evidence/outcome-adaptive-per-arm-c-tmle-parametric.md"
    ),
    anchor="outcome-adaptive-per-arm-c-tmle-parametric",
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
        "tests/studies/canonical_ctmle_oat_per_arm_parametric.py",
        "tests/studies/ctmle_oat_per_arm_parametric_properties.py",
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
    runner_module="tests.studies.canonical_ctmle_oat_per_arm_parametric",
    properties_module="tests.studies.ctmle_oat_per_arm_parametric_properties",
    property_cells=property_cells(),
    publication_policy="gated",
    calibration_efficiency_ratio=False,
)

CONFIGURATION: dict[str, Any] = {
    "strategy": "oat",
    "oat_design": "per_arm",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "outcome_learner": (
        "ParametricOutcome: unpenalized logistic GLM on [A indicators, W, A x W] "
        "(quasi-binomial on the scaled outcome when q_bounds is declared)"
    ),
    "treatment_learner": (
        "ParametricMechanism: per arm, unpenalized logistic regression of 1{A = a} on the "
        "cubic in l = logit(clip(Qbar(a, W), 1e-6, 1 - 1e-6)); the kappa cell also fits the "
        "linear basis"
    ),
    "g_bounds": list(laws.G_BOUNDS),
    "g_bounds_by_cell": {
        "bounds_active": list(STEEP_BOUNDS),
        "auto_bounds": "auto",
    },
    "q_bounds_by_cell": {"bounded_outcome": list(BOUNDED_OUTCOME)},
    "covariates_by_cell": {"pseudo_true": ["W2", "W3"]},
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
        "curve; R parity of the point estimate carries through test_parametric_learners.py"
    ),
    "budgets": dict(BUDGETS),
    "budget_rule": (
        "predictive rule with tau_80 = 0.0080 pooled over the study's gated rows; each row "
        "centred on its own pilot SE ratio with variance s^2 + tau_80^2; each of the 12 gated "
        "calibration groups sized to the smallest multiple of 400 with pass probability >= "
        "0.95 ** (1 / 12), so the study passes with probability >= 0.95; sharp_null and power "
        "are sized in this module's BUDGETS comment"
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
        "finds no defect). shipped_curve_control and cross_term_dropped_control must be red, "
        "that is, their SE-ratio intervals must leave the calibration band below and above. "
        "The harness publishes passed=True for such a control. If either control stays inside "
        "or straddles the band, its cell cannot see its term, the study is invalid, and it is "
        "redesigned before any reading. The coverage of psi_0 in the pseudo-true cell, the "
        "sqrt(n) SD of the ladder and the Monte Carlo covariance of the arm means are reported "
        "and gate nothing."
    ),
    "negative_controls": dict(NEGATIVE_CONTROLS),
    "negative_control_expected_se_ratio": {
        "shipped_curve_control": 0.845,
        "cross_term_dropped_control": 1.228,
    },
    "publication_reading": (
        "gated: every row must pass, and a negative control passes by being red"
    ),
    "family_mapping": (
        "the ladder and the joint cells are interval_calibration cells named "
        "<label>__<estimand>__corrected, so the shared calibration verdict applies per estimand; "
        "the covariance is reported on the shared-prognosis ate row"
    ),
    "failure_probe": (
        "pilot of 5,000 draws per cell (plan section 11): no separation refusal, no stack "
        "failure, largest recomputed-Q* gap 4.4e-16"
    ),
}


# ---------------------------------------------------------------------------------- the laws


def sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of ``law`` (columns ``W1``, ``W2``, ``W3``, ``A``, ``Y``)."""
    if law == "binary_active":
        return laws.BINARY_ACTIVE.sample(n, seed)[0]
    if law == "binary_null":
        return laws.BINARY_NULL.sample(n, seed)[0]
    if law == "shared_prognosis":
        return quad.sample(quad.SHARED_PROGNOSIS, n, seed)
    if law == "u_shaped":
        return quad.sample(quad.U_SHAPED, n, seed)
    if law == "steep":
        return quad.sample(quad.STEEP, n, seed)
    if law == "bounded":
        return quad.sample(quad.BINARY_ACTIVE, n, seed, bounded=BOUNDED_OUTCOME)
    raise KeyError(law)


@cache
def arm_means(law: str) -> tuple[float, float]:
    """``(psi_0, psi_1)`` of ``law``: the true arm means, or the limit ``psi*`` for ``pseudo_true``.

    ``pseudo_true`` is ``binary_active`` fitted without ``W1`` in the outcome features, whose
    limit is the quadrature's ``psi*``.  ``bounded`` is ``2 + 8 psi`` of ``binary_active``.
    """
    if law in ("binary_active", "binary_null"):
        truth = (laws.BINARY_ACTIVE if law == "binary_active" else laws.BINARY_NULL).truth()
        return float(truth["ey0"]), float(truth["ey1"])
    if law == "bounded":
        low, high = arm_means("binary_active")
        lower, upper = BOUNDED_OUTCOME
        return lower + (upper - lower) * low, lower + (upper - lower) * high
    if law == "pseudo_true":
        star = quad.population_fit(quad.BINARY_ACTIVE, covariates=(1, 2)).psi_star
        return float(star[0]), float(star[1])
    fits = {
        "shared_prognosis": lambda: quad.population_fit(quad.SHARED_PROGNOSIS),
        "u_shaped": lambda: quad.population_fit(quad.U_SHAPED),
        "steep": lambda: quad.population_fit(quad.STEEP, nodes=50, bounds=STEEP_BOUNDS),
    }
    psi = fits[law]().psi_0
    return float(psi[0]), float(psi[1])


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
    control and the reported ``sqrt(n)`` SD of the ladder.
    """
    fit = quad.population_fit(quad.BINARY_ACTIVE)
    variance = quad.covariance(fit)
    return {
        "ey0": float(np.sqrt(variance[0, 0])),
        "ey1": float(np.sqrt(variance[1, 1])),
        "ate": quad.ate_sd(fit),
    }


# ---------------------------------------------------------------------------------- the fit


def estimator(fit: Fit) -> CTMLE:
    """The subject of one configuration: a per-arm ``CTMLE`` in sample, with the parametric learners."""
    outcome_learner = (
        ParametricOutcome() if fit.outcome == "parametric" else laws.OracleOutcome("binary_active")
    )
    options: dict[str, Any] = {} if fit.q_bounds is None else {"q_bounds": fit.q_bounds}
    return CTMLE(
        strategy="oat",
        oat_design="per_arm",
        outcome_learner=outcome_learner,
        treatment_learner=ParametricMechanism(degree=fit.degree),
        cross_fit=False,
        simultaneous=False,
        estimands=fit.estimands,
        g_bounds=fit.g_bounds,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        **options,
    )


def fit_cleverly(frame: pd.DataFrame, fit: Fit) -> Any:
    """Fit the study's subject on one sample.

    An admitted fit must report ``"influence_curve"`` for every requested estimand.  A
    ``"stack_failure_plugin"`` estimate is a failure of the replicate: the study measures the
    stacked curve, and a fall back to the shipped curve would measure a different estimator.
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
