"""Registered evidence for the ``K``-arm, weighted and population-target stacked per-arm C-TMLE curve.

This is X58 part (b) and part (d), X58b2.  The subject is
``CTMLE(strategy="oat", oat_design="per_arm")`` with :class:`~cleverly.learners.ParametricOutcome`
and :class:`~cleverly.learners.ParametricMechanism`, in sample and cross-fitted, with three arms,
with known observation weights, and for the targets ``ey_obs``, ``par`` and ``paf`` and the
simultaneous band.  Each such fit is a finite-dimensional stacked Z-estimator
(:mod:`cleverly.estimators._oat_stack`; plan 13.2 and 13.4), and an admitted estimate carries the
status ``"influence_curve"``.  The study ``ctmle-oat-per-arm-parametric`` (X58a) measures the
binary in-sample form, and ``ctmle-oat-per-arm-parametric-crossfit`` (X58b1) the binary
cross-fitted form.

**No comparator.**  ``drtmle``'s ``adapt_g`` reports the shipped curve, so no R program measures
the corrected uncertainty.  The study has ``reference=None``, runs no R, and its evidence page
labels the interval "Python property evidence, no comparator".

**The laws** are those of plan 13.6, all in :mod:`tests.studies.oat_per_arm_laws`:
``binary_active``, ``binary_active_weighted`` (truth on the half-line grid),
``three_arm_active``, ``three_arm_shared`` (the strong shared prognosis, arm correlations 0.25 to
0.26), ``three_arm_active_weighted`` and ``three_arm_null`` (every arm at ``c = -0.3`` and
``u = (0.8, -0.8)``).  Every truth is a quadrature, and the reference arm is per group: the
in-sample ``three_arm`` group, ``three_arm_shared``, ``composition`` and ``three_arm_null`` fit
``reference="1"``, and the groups that re-measure the old cross-fitted three-arm cells fit
``reference="0"``.

**The primary phase** fits ``binary_active`` for ``par`` and ``paf`` and ``three_arm_active``
(``reference="0"``) for every estimand of ``canonical_ctmle_oat_per_arm.py``, in sample at
``n = 1,500``, for 1,600 replicates per law.  These are the old in-sample primary rows that X58a
did not take.  The driver gates them, so ``G`` counts the phase as a unit.  Every other cell is a
property cell of :mod:`tests.studies.ctmle_oat_per_arm_parametric_extensions_properties`.

**The declared reading** (plan 13.6, 5.4 carried over), fixed before any run:

* A red calibration row is a finite-sample result only under the AGENT-BRIEF rule: its 99%
  SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no
  defect.  Any other red row is a defect suspect, and it is diagnosed before anything is
  published.
* The nine :data:`NEGATIVE_CONTROLS` are negative controls.  Each *must* be red on its declared
  side.  A control row publishes ``passed = True`` exactly when its SE-ratio interval leaves the
  calibration band on that side, or, for a joint control, when the upper end of its
  joint-coverage interval is below 0.95.  A control that stays inside the band, or straddles it,
  publishes ``passed = False``.  Its cell then cannot see its term, the study is invalid, and it
  is redesigned before any reading.  The policy is ``gated``, so such a row blocks publication.
* The exact-law values of the controls are in ``CONFIGURATION["negative_control_expected"]``.
  The joint-coverage figures are for five three-arm estimates: 0.826 (pointwise, either
  reference), 0.861 (shipped band, ``reference="1"``, ``three_arm``) and 0.867 (shipped band,
  ``reference="0"``, ``three_arm_crossfit``).  For the six estimates of the ``binary_population``
  band set the exact values are 0.841 (pointwise) and 0.872 (shipped band).  The stacked band
  covers 0.950 on each declared band set.
* The ``ey_obs`` rows of ``three_arm``, ``binary_population`` and ``weighted`` are gated positive
  cells.  Their curve is the shipped empirical one, and the support rows ``C11.binary``,
  ``C11.k_arms`` and ``C11.weighted`` license the interval of ``ey_obs`` (family ``P4``), so a
  measured cell backs each of those claims (design review m2).  The Monte Carlo covariance of the
  two arm means against the mean reported covariance in ``three_arm_shared`` is reported and gates
  nothing.  The paired deficit of the generated-design pair is reported and gates nothing.

**Failures.**  The record declares ``failure_bound={IMPLEMENTATION: 0.01}`` and the property
module declares ``FAILURE_BOUND = 0.01``, in the form ``docs/development/method-benchmarking.md``
("Property fitting failures", "The bound") defines.  A failed replicate is recorded and not
redrawn, and the run continues.  A fit that raises, and an admitted fit with an estimate whose
status is not ``"influence_curve"`` (``fit_cleverly`` raises on it), both count as failed
replicates.  A band cell therefore counts a replicate as lost when any estimate of its declared
band set has another status (then the band covers a subset or is not built, plan D3).  A cell may
lose ``floor(0.01 R)`` replicates of its ``R``, each clause is judged by its worst case over the
lost values, and one more lost replicate reads ``bound exceeded``.

**Sizes.**  Plan 13.6 sizes the study by the study-wise rule of section 12.  ``G`` counts nineteen
units: eight calibration groups (the generated-design pair among them), one type I group, the
primary phase and nine negative controls.  Each unit is sized so that its pass probability is at
least ``0.95 ** (1 / 19) = 0.99730``, so the study passes with probability at least 0.95 when
nothing is wrong.  Pilot step 2 reads every group on pilot-only seeds, and the budgets of
:data:`BUDGETS` are set from it and from step 4 by the rule in ``CONFIGURATION["budget_rule"]``.
Pilot step 4 re-reads each group at its declared budget on fresh pilot-only seeds and reports the
study-wise pass probability before any declared run.  A budget moves only by a later commit that
cites a pilot.
"""

from __future__ import annotations

import math
import re
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
from tests.studies import oat_multi_arm_quadrature as mq
from tests.studies import oat_per_arm_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_study_helpers import primary_rows

#: Replicates per law of the primary phase: two laws, 3,200 replicates in all (plan 13.6).  The
#: smallest multiple of 400 at the per-unit target, on 11 independent rows at a true coverage of
#: 0.95, is 1,200.  1,600 is kept (design review m6) because it holds the target down to a true
#: coverage of 0.945 on every row (0.99990, where 1,200 gives 0.99478) and stays at 0.99592 at
#: 0.94 (1,200 gives 0.93764).  The three-arm in-sample
#: ``rr`` and ``or`` rows have no earlier measurement under the stacked curve.  The cost is
#: 0.04 CPU-h.
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 1_500
#: Fresh entropy: no registered study uses either number, and
#: ``tests/unit/test_ctmle_oat_per_arm_parametric_extensions_study_design.py`` checks that.
SEED = 20261178
RESAMPLING_SEED = 2026117801
IMPLEMENTATION = "cleverly-ctmle-oat-per-arm-parametric-extensions"
#: The share of a cell's replicates that a run may lose, declared before the first run.  The
#: record carries it as ``failure_bound={IMPLEMENTATION: FAILURE_BOUND}`` and the property module
#: declares the same value.
FAILURE_BOUND = 0.01
#: The factor of ``shrunken_se_control``.  The shared ``calibration_controls`` value.
SHRUNKEN_SE_FACTOR = 0.70
#: The fold count of every cross-fitted group.  The support rows ``C11.stacked_k_arms``,
#: ``C11.weighted_stacked`` and ``C11.stacked_k_arms_weighted`` state it as the measured value.
FOLDS = 5

#: The primary scenario -> the estimands it transcribes.
BINARY_PRIMARY = ("par", "paf")
THREE_ARM_PRIMARY = (
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
SCENARIOS: Mapping[str, tuple[str, ...]] = {
    "binary_active": BINARY_PRIMARY,
    "three_arm_active": THREE_ARM_PRIMARY,
}

# ---------------------------------------------------------------------------------- the sizes

#: The replicates of each fit group, each a multiple of 400.  The plan 13.6 (rev 3.1) expected
#: budgets stand for the groups the pilot did not raise.  Three are raised from the pilot (step 2
#: at 5,000 draws and step 4 at the declared size, on independent seeds, pooled):
#: ``binary_population`` 9,600 to 11,200 and ``three_arm_crossfit`` 17,600 to 18,800 are the
#: smallest multiples of 400 whose predicted pass
#: probability reaches the per-unit target.  ``three_arm_generated_design`` reaches 0.9932 at
#: 80,000 and no budget up to 80,000 reaches the target (see ``budget_rule``).  The eight
#: calibration groups are sized to the per-unit target ``0.95 ** (1 / 19)``.
#: ``three_arm_null`` is sized at 1,600 (design review m1): at most 129 rejections pass the type I
#: verdict.  With every allowed lost replicate (16) counted as a rejection, the two rows pass with
#: probability 0.99982 at a true rate of 0.05 and 0.99490 at 0.055.  At 1,200 the same worst case
#: gave 0.99506 and 0.95525.
BUDGETS: Mapping[str, int] = {
    "three_arm": 29_200,
    "three_arm_shared": 14_000,
    "binary_population": 11_200,
    "three_arm_crossfit": 18_800,
    "three_arm_generated_design": 80_000,
    "weighted": 17_600,
    "weighted_crossfit": 11_600,
    "composition": 17_600,
    "three_arm_null": 1_600,
}
#: The units of the study-wise rule (plan 13.6): calibration groups, type I groups, the primary
#: phase and the declared negative controls.
UNITS: Mapping[str, int] = {
    "calibration_groups": 8,
    "type_i_groups": 1,
    "primary_phase": 1,
    "negative_controls": 9,
}
UNIT_COUNT = sum(UNITS.values())
NULL_N = 1_000
CALIBRATION_N = 1_500


# ------------------------------------------------------------------------------ the learners


class ShippedOutcome(ParametricOutcome):
    """A :class:`~cleverly.learners.ParametricOutcome` that the support predicate refuses.

    The predicate admits exactly ``ParametricOutcome``, so a subclass keeps the shipped
    adaptive-propensity curve and its plug-in status.  The fit is otherwise identical, so the
    shipped band control reads the band the shipped curves give on the same estimates.
    """


@dataclass(frozen=True)
class Fit:
    """One ``CTMLE`` configuration of the study.

    Parameters
    ----------
    name : str
        The configuration's name inside its group.
    estimands : tuple of str
        The estimands the fit requests.
    k : int
        The number of arms of the law.
    outcome : {"parametric", "oracle", "shipped"}
        ``"parametric"`` fits :class:`~cleverly.learners.ParametricOutcome`, which the support
        predicate admits.  ``"oracle"`` pins ``Qbar_0`` of ``three_arm_active``, which the
        predicate refuses.  ``"shipped"`` fits :class:`ShippedOutcome`, which it refuses too.  A
        refused fit keeps the shipped curve and its plug-in status.
    cross_fit : bool
        Whether the fit is cross-fitted over :data:`FOLDS` folds.
    reference : str or None
        The reference arm label of a three-arm fit, or ``None`` at two arms.
    weighted : bool
        Whether the fit reads the weight column ``wt`` of the law.
    simultaneous : bool
        Whether the fit builds the simultaneous band.
    covariates : tuple of str
        The covariate columns.
    """

    name: str
    estimands: tuple[str, ...]
    k: int = 2
    outcome: str = "parametric"
    cross_fit: bool = False
    reference: str | None = None
    weighted: bool = False
    simultaneous: bool = False
    covariates: tuple[str, ...] = ("W1", "W2", "W3")


@dataclass(frozen=True)
class Band:
    """One simultaneous band cell, read from one fit.

    Parameters
    ----------
    label : str
        The cell prefix: ``<label>__simultaneous_band``.
    fit : str
        The fit whose band the cell reads.
    names : tuple of str
        The declared band set: the estimates the band must cover, exactly.
    pointwise : bool
        Whether the group also declares ``<label>__pointwise_joint_control``.
    shipped : str or None
        The ``"shipped"`` fit whose estimates, banded as the stacked ones are, give
        ``<label>__shipped_curve_band_control``, or ``None`` for no such control.
    """

    label: str
    fit: str
    names: tuple[str, ...]
    pointwise: bool = False
    shipped: str | None = None


@dataclass(frozen=True)
class Group:
    """The cells that share one sample per replicate and the fits made on it.

    Parameters
    ----------
    name : str
        The group's name; it labels the checkpoints and the sample stream.
    law : str
        The name of the law the sample is drawn from (:data:`~tests.studies.oat_per_arm_laws.LAWS`).
    n : int
        The sample size.
    fits : tuple of Fit
        The configurations fitted on each sample.
    bands : tuple of Band
        The simultaneous band cells of the group.
    """

    name: str
    law: str
    n: int
    fits: tuple[Fit, ...]
    bands: tuple[Band, ...] = ()

    @property
    def replicates(self) -> int:
        """The group's budget, :data:`BUDGETS`."""
        return BUDGETS[self.name]


ARMS3 = ("ey", "ate")
POPULATION = ("ey0", "ey1", "ate", "ey_obs", "par", "paf")
BINARY_FULL = ("ey0", "ey1", "ate", "rr", "or", "ey_obs", "par", "paf")
THREE_FULL = ("ey", "ate", "rr", "or", "ey_obs", "par", "paf")

#: The five estimates of the three-arm band sets, by reference.
BAND_REF1 = ("ey[0]", "ey[1]", "ey[2]", "ate[0 vs 1]", "ate[2 vs 1]")
BAND_REF0 = ("ey[0]", "ey[1]", "ey[2]", "ate[1 vs 0]", "ate[2 vs 0]")

GROUPS: Mapping[str, Group] = {
    group.name: group
    for group in (
        Group(
            "three_arm",
            "three_arm_active",
            CALIBRATION_N,
            (
                Fit("main", THREE_FULL, k=3, reference="1"),
                Fit("band", ARMS3, k=3, reference="1", simultaneous=True),
                Fit("shipped_band", ARMS3, k=3, reference="1", outcome="shipped"),
            ),
            (Band("three_arm", "band", BAND_REF1, pointwise=True, shipped="shipped_band"),),
        ),
        Group(
            "three_arm_shared",
            "three_arm_shared",
            CALIBRATION_N,
            (Fit("main", ARMS3, k=3, reference="1"),),
        ),
        Group(
            "binary_population",
            "binary_active",
            CALIBRATION_N,
            (
                Fit("main", POPULATION, simultaneous=True),
                Fit("shipped_band", POPULATION, outcome="shipped"),
            ),
            (
                Band(
                    "binary_population", "main", POPULATION, pointwise=True, shipped="shipped_band"
                ),
            ),
        ),
        Group(
            "three_arm_crossfit",
            "three_arm_active",
            CALIBRATION_N,
            (Fit("main", ARMS3, k=3, cross_fit=True, reference="0", simultaneous=True),),
            (Band("three_arm_crossfit", "main", BAND_REF0),),
        ),
        Group(
            "three_arm_generated_design",
            "three_arm_active",
            CALIBRATION_N,
            (
                Fit("estimated", ("ate",), k=3, cross_fit=True, reference="0"),
                Fit("oracle", ("ate",), k=3, cross_fit=True, reference="0", outcome="oracle"),
            ),
        ),
        Group(
            "weighted",
            "binary_active_weighted",
            CALIBRATION_N,
            (Fit("main", BINARY_FULL, weighted=True),),
        ),
        Group(
            "weighted_crossfit",
            "binary_active_weighted",
            CALIBRATION_N,
            (Fit("main", ("ey0", "ey1", "ate"), cross_fit=True, weighted=True),),
        ),
        Group(
            "composition",
            "three_arm_active_weighted",
            CALIBRATION_N,
            (Fit("main", ARMS3, k=3, cross_fit=True, reference="1", weighted=True),),
        ),
        Group(
            "three_arm_null",
            "three_arm_null",
            NULL_N,
            (Fit("main", ("ate",), k=3, reference="1"),),
        ),
    )
}


# ---------------------------------------------------------------------------------- the cells


def slug(token: str) -> str:
    """The cell-name form of an estimate name: ``ate[0 vs 1]`` is ``ate_0_vs_1``."""
    return token.replace("[", "_").replace("]", "").replace(" vs ", "_vs_")


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
    estimand : str
        The estimate name the cell reads, with ``log_`` before a ratio read on the log scale
        (``log_rr[0 vs 1]``), or ``max_t`` for a joint cell.
    truth : str
        The name of the law whose truth defines the cell (:func:`truth`).
    recipe : str
        How the row is built.  ``"corrected"`` reads the estimate and its standard error.
        ``"shipped_curve"`` reads the standard error of the shipped adaptive-propensity curve of
        the same fit, stored per draw.  ``"cross_term_dropped"`` reads
        ``sqrt(se_a^2 + se_b^2)`` of the two arm means of an ``ate`` row.  ``"band"``,
        ``"pointwise"`` and ``"shipped_band"`` are the three joint rows of a :class:`Band`.
    role : {"positive", "control"}
        The row's role.
    report : str
        A quantity the row also records and no verdict reads.  ``"covariance"``: both arm
        estimates and the reported covariance.
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


JOINT = "simultaneous_coverage"


def _calibration(
    group: str,
    tokens: tuple[str, ...],
    *,
    fit: str = "main",
    report: Mapping[str, str] | None = None,
) -> list[Cell]:
    """The positive ``<group>__<estimate>__corrected`` cells of one fit."""
    law = GROUPS[group].law
    return [
        Cell(
            group,
            "interval_calibration",
            f"{group}__{slug(token)}__corrected",
            group,
            fit,
            token,
            law,
            report=(report or {}).get(token, ""),
        )
        for token in tokens
    ]


def _control(group: str, token: str, recipe: str, kind: str, *, fit: str = "main") -> Cell:
    """A declared SE control of ``token``: ``<group>__<estimate>__<recipe>__<kind>``."""
    return Cell(
        group,
        "interval_calibration",
        f"{group}__{slug(token)}__{recipe}__{kind}",
        group,
        fit,
        token,
        GROUPS[group].law,
        recipe=recipe,
        role="control",
    )


def _joint(group: str, band: Band) -> list[Cell]:
    """The band cell of ``band``, then its pointwise control and its shipped band control."""
    law = GROUPS[group].law
    out = [
        Cell(
            group, JOINT, f"{band.label}__simultaneous_band", group, band.fit, "max_t", law, "band"
        )
    ]
    if band.pointwise:
        out.append(
            Cell(
                group,
                JOINT,
                f"{band.label}__pointwise_joint_control",
                group,
                band.fit,
                "max_t",
                law,
                "pointwise",
                "control",
            )
        )
    if band.shipped is not None:
        out.append(
            Cell(
                group,
                JOINT,
                f"{band.label}__shipped_curve_band_control",
                group,
                band.shipped,
                "max_t",
                law,
                "shipped_band",
                "control",
            )
        )
    return out


THREE_REF1_ROWS = (
    "ey[0]",
    "ey[1]",
    "ey[2]",
    "ate[0 vs 1]",
    "ate[2 vs 1]",
    "log_rr[0 vs 1]",
    "log_rr[2 vs 1]",
    "log_or[0 vs 1]",
    "log_or[2 vs 1]",
    "par[1]",
    "paf[1]",
    "ey_obs",
)
BINARY_WEIGHTED_ROWS = ("ey0", "ey1", "ate", "log_rr", "log_or", "par", "paf", "ey_obs")


def _cells() -> tuple[Cell, ...]:
    out: list[Cell] = []
    out += _calibration("three_arm", THREE_REF1_ROWS)
    out.append(_control("three_arm", "ey[1]", "shipped_curve", "shrunken_se_control"))
    out += _joint("three_arm", GROUPS["three_arm"].bands[0])
    out += _calibration(
        "three_arm_shared",
        BAND_REF1,
        report={"ate[0 vs 1]": "covariance"},
    )
    out.append(
        _control("three_arm_shared", "ate[0 vs 1]", "cross_term_dropped", "inflated_se_control")
    )
    out += _calibration("binary_population", ("par", "paf", "ey_obs"))
    out.append(_control("binary_population", "par", "shipped_curve", "shrunken_se_control"))
    out += _joint("binary_population", GROUPS["binary_population"].bands[0])
    out += _calibration("three_arm_crossfit", BAND_REF0)
    out.append(_control("three_arm_crossfit", "ey[1]", "shipped_curve", "shrunken_se_control"))
    out += _joint("three_arm_crossfit", GROUPS["three_arm_crossfit"].bands[0])
    out += [
        Cell(
            "three_arm_generated_design",
            "generated_design",
            f"{GENERATED_LABEL}__{kind}__{slug(contrast)}",
            "three_arm_generated_design",
            fit,
            contrast,
            "three_arm_active",
        )
        for contrast in GENERATED_CONTRASTS
        for kind, fit in (("oracle_design", "oracle"), ("estimated", "estimated"))
    ]
    out += _calibration("weighted", BINARY_WEIGHTED_ROWS)
    out.append(_control("weighted", "ey0", "shipped_curve", "shrunken_se_control"))
    out += _calibration("weighted_crossfit", ("ey0", "ey1", "ate"))
    out += _calibration("composition", BAND_REF1)
    out += [
        Cell(
            "three_arm_null",
            "type_i_error",
            f"three_arm_null__{slug(contrast)}",
            "three_arm_null",
            "main",
            contrast,
            "three_arm_null",
        )
        for contrast in ("ate[0 vs 1]", "ate[2 vs 1]")
    ]
    return tuple(out)


#: The label of the generated-design pair's cells: ``three_arm_active__oracle_design__ate_1_vs_0``
#: and ``three_arm_active__estimated__ate_1_vs_0``, both cross-fitted, ``reference="0"``.
GENERATED_LABEL = "three_arm_active"
GENERATED_CONTRASTS = ("ate[1 vs 0]", "ate[2 vs 0]")

CELLS: tuple[Cell, ...] = _cells()

#: The calibration cells that the harness derives two controls from, without a fit: the
#: ``shrunken_se_control`` and the ``noise_control`` of each group's contrast row.  A group with
#: no gated ``ate`` row (``binary_population``) derives them from its ``par`` row.
CONTROL_LABELS: tuple[str, ...] = (
    "three_arm__ate_0_vs_1",
    "three_arm_shared__ate_0_vs_1",
    "binary_population__par",
    "three_arm_crossfit__ate_1_vs_0",
    "weighted__ate",
    "weighted_crossfit__ate",
    "composition__ate_0_vs_1",
)
DERIVED_CONTROLS = tuple(
    f"{label}__{kind}"
    for label in CONTROL_LABELS
    for kind in ("shrunken_se_control", "noise_control")
)

#: The groups of plan 13.6 for this study, each with the group it reads.  The design test pins
#: the list against the plan.
PLAN_CELLS: Mapping[str, str] = {name: name for name in BUDGETS}

#: The negative controls that must be red (plan 13.6), by published cell, with the side.  A joint
#: control is red when the upper end of its joint-coverage interval is below 0.95.
NEGATIVE_CONTROLS: Mapping[str, str] = {
    "three_arm__ey_1__shipped_curve__shrunken_se_control": "below",
    "three_arm__pointwise_joint_control": "below",
    "three_arm__shipped_curve_band_control": "below",
    "three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control": "above",
    "binary_population__par__shipped_curve__shrunken_se_control": "below",
    "binary_population__pointwise_joint_control": "below",
    "binary_population__shipped_curve_band_control": "below",
    "three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control": "below",
    "weighted__ey0__shipped_curve__shrunken_se_control": "below",
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
    name="K-arm, weighted and population-target outcome-adaptive per-arm C-TMLE with a parametric outcome",
    slug="ctmle-oat-per-arm-parametric-extensions",
    artifacts=ROOT / "tests" / "canonical" / "ctmle_oat_per_arm_parametric_extensions",
    document=(
        "docs/technical-reference/method-evidence/"
        "outcome-adaptive-per-arm-c-tmle-parametric-extensions.md"
    ),
    anchor="outcome-adaptive-per-arm-c-tmle-parametric-extensions",
    scenarios=dict(SCENARIOS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
    modules=(
        "tests/studies/canonical_ctmle_oat_per_arm_parametric_extensions.py",
        "tests/studies/ctmle_oat_per_arm_parametric_extensions_properties.py",
        "tests/studies/oat_multi_arm_quadrature.py",
        "tests/studies/oat_parametric_quadrature.py",
        "tests/studies/oat_per_arm_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_ctmle_oat_per_arm_parametric_extensions",
    properties_module="tests.studies.ctmle_oat_per_arm_parametric_extensions_properties",
    property_cells=property_cells(),
    publication_policy="gated",
    calibration_efficiency_ratio=False,
)

CONFIGURATION: dict[str, Any] = {
    "strategy": "oat",
    "oat_design": "per_arm",
    "n_folds_cross_fitted": FOLDS,
    "repeats": 1,
    "simultaneous_intervals": "only the band fits of three_arm, binary_population and three_arm_crossfit",
    "outcome_learner": (
        "ParametricOutcome(arm_columns=K - 1): unpenalized logistic GLM on [A indicators, W, "
        "A x W]; one fit per fold when cross-fitted"
    ),
    "treatment_learner": (
        "ParametricMechanism: per arm (and fold), unpenalized logistic regression of 1{A = a} on "
        "the cubic in l = logit(clip(Qbar_k(a, W), 1e-6, 1 - 1e-6))"
    ),
    "g_bounds": list(laws.G_BOUNDS),
    "max_iter": 100,
    "tol": 1e-10,
    "random_state": 0,
    "reference_by_group": {
        group.name: next((fit.reference for fit in group.fits if fit.reference), None)
        for group in GROUPS.values()
    },
    "weights": "wt = 0.5 + 1{W1 > 0}, known, on binary_active_weighted and three_arm_active_weighted",
    "band_sets": {
        band.label: list(band.names) for group in GROUPS.values() for band in group.bands
    },
    "primary": {
        "scenarios": {name: list(estimands) for name, estimands in SCENARIOS.items()},
        "n": PRIMARY_N,
        "replicates_per_law": PRIMARY_REPLICATES,
        "fit": "in sample; three_arm_active at reference 0",
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
        "plan section 12 with G = 19 gated units, each sized to pass with probability >= "
        "0.95 ** (1 / 19) = 0.99730; binary_population (11,200) and three_arm_crossfit (18,800) are "
        "raised to the smallest multiple of 400 that reaches the target on the pooled pilot of steps "
        "2 and 4; three_arm_generated_design is set to 80,000, the largest size searched, where the "
        "predicted pass probability is 0.9932 and still below the target (the pooled tau_80 of 0.0106 "
        "applied to oracle-design ratios near 1.03 caps it); three_arm_null is sized at 1,600 (at most 129 rejections "
        "pass, and the target holds at every allowed loss count at a true rate of 0.05); the "
        "primary phase runs 1,600 replicates per law, which holds the target down to a true "
        "coverage of 0.945 on every row and gives 0.99592 at 0.94, where the smallest size at "
        "0.95 is 1,200 and gives 0.93764 at 0.94"
    ),
    "failure_bounds": {
        "declared": FAILURE_BOUND,
        "applied": (
            "0.01 in both phases: a cell may lose floor(0.01 R) replicates, and each clause "
            "is judged by its worst case over the lost values; one more lost replicate reads "
            "bound exceeded. A fit that raises, and an admitted fit with an estimate whose status "
            "is not influence_curve, both count as failed replicates"
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
        "finds no defect). The nine negative controls must be red on their declared side: the "
        "SE-ratio interval of a shipped-curve control leaves the calibration band below, that of "
        "the cross_term_dropped control leaves it above, and the joint-coverage interval of a "
        "pointwise or shipped-band control lies below 0.95. The harness publishes passed=True "
        "for such a control. If one stays inside or straddles its threshold, its cell cannot see "
        "its term, the study is invalid, and it is redesigned before any reading. The "
        "Monte Carlo covariance of the two arm means of three_arm_shared against the "
        "mean reported covariance and the paired deficit of the generated-design pair are "
        "reported and gate nothing."
    ),
    "negative_controls": dict(NEGATIVE_CONTROLS),
    # The exact-law figures of plan 13.2 and 13.6 (the quadrature logs).  The shipped ratios are
    # the SD of the shipped curve over the stacked SD; a joint figure is the probability that the
    # stacked estimates all fall inside the band of the named curve.  The design test recomputes
    # each on its declared band set, and the stacked band covers 0.9505 (three_arm), 0.9497
    # (three_arm_crossfit) and 0.950 (binary_population).
    "negative_control_expected": {
        "three_arm__ey_1__shipped_curve__shrunken_se_control": {"se_ratio": 0.780},
        "three_arm__pointwise_joint_control": {"joint_coverage": 0.826},
        "three_arm__shipped_curve_band_control": {"joint_coverage": 0.861},
        "three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control": {
            "se_ratio": 1.163
        },
        "binary_population__par__shipped_curve__shrunken_se_control": {"se_ratio": 0.766},
        "binary_population__pointwise_joint_control": {"joint_coverage": 0.841},
        "binary_population__shipped_curve_band_control": {"joint_coverage": 0.872},
        "three_arm_crossfit__ey_1__shipped_curve__shrunken_se_control": {"se_ratio": 0.780},
        "weighted__ey0__shipped_curve__shrunken_se_control": {"se_ratio": 0.819},
    },
    "publication_reading": (
        "gated: every row must pass, and a negative control passes by being red"
    ),
    "family_mapping": (
        "the groups are interval_calibration cells named <group>__<estimate>__corrected, so "
        "the shared calibration verdict applies per estimate; the bands are simultaneous_coverage "
        "cells"
    ),
}


# ---------------------------------------------------------------------------------- the laws


def sample(law: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of ``law`` (columns ``W1``, ``W2``, ``W3``, ``A``, ``Y``, and ``wt`` if weighted)."""
    return laws.LAWS[law].sample(n, seed)[0]


@cache
def _law_truth(law: str) -> tuple[tuple[float, ...], float]:
    """``(arm means, E[Y])`` of ``law``, under the weight-tilted law when it is weighted."""
    spec = laws.LAWS[law]
    return tuple(float(value) for value in spec.arm_means()), float(spec.observed_mean())


_NAME = re.compile(r"^(?P<log>log_)?(?P<name>ey_obs|ey|ate|rr|or|par|paf)(?P<arm>\d|\[.+\])?$")


def truth(law: str, estimand: str) -> float:
    """The truth of ``estimand`` on ``law``; ``log_rr`` and ``log_or`` are on the log scale.

    The estimate name follows the fit: ``ey0``, ``ate``, ``par`` at two arms (reference arm 0),
    and ``ey[1]``, ``ate[0 vs 1]``, ``par[1]`` at three.  ``ey_obs`` is ``E[Y]``.
    """
    match = _NAME.match(estimand)
    if match is None:
        raise KeyError(estimand)
    means, observed = _law_truth(law)
    name, arm = match["name"], match["arm"]
    if name == "ey_obs":
        return observed
    if name == "ey":
        if arm is None:
            raise KeyError(estimand)
        return means[int(arm.strip("[]"))]
    if name in ("par", "paf"):
        low = means[int(arm.strip("[]"))] if arm else means[0]
        value = observed - low if name == "par" else 1.0 - low / observed
    else:
        first, second = (int(part) for part in arm[1:-1].split(" vs ")) if arm else (1, 0)
        high, low = means[first], means[second]
        value = {"ate": high - low, "rr": high / low, "or": _odds(high) / _odds(low)}[name]
    return math.log(value) if match["log"] else value


def _odds(p: float) -> float:
    return p / (1.0 - p)


_ARM_LAWS: Mapping[str, Any] = {
    "binary_active": lambda: mq.BINARY_ACTIVE,
    "binary_active_weighted": lambda: mq.BINARY_ACTIVE_WEIGHTED,
    "three_arm_active": lambda: mq.THREE_ARM_ACTIVE,
    "three_arm_shared": lambda: mq.THREE_ARM_SHARED,
    "three_arm_active_weighted": lambda: mq.from_oat_law(laws.THREE_ARM_ACTIVE_WEIGHTED),
    "three_arm_null": lambda: mq.from_oat_law(laws.THREE_ARM_NULL),
}


@cache
def exact_sds(law: str, reference: int = 0) -> dict[str, float]:
    """The asymptotic ``sqrt(n)`` SD of the stacked curve of each estimate, by name.

    The names are those of :func:`truth` for a ``reference`` arm: ``ey[a]``, ``ate[a vs r]``,
    ``log_rr[a vs r]``, ``par[r]`` and ``paf[r]``, and at two arms the same with ``ey0``,
    ``ate``, ``log_rr``, ``par``, ``paf``.  The quadrature is
    :mod:`tests.studies.oat_multi_arm_quadrature`, whose curves are the production code's.
    """
    arm_law = _ARM_LAWS[law]()
    nodes = mq.half_line_nodes(60, 20) if arm_law.weight is not None else mq.hermite_nodes(20)
    fit = mq.population_fit(arm_law, *nodes)
    named = mq.contrast_curves(fit, mq.arm_curves(fit), reference)
    out = {name: mq.sd(fit, curve) for name, curve in named.items()}
    if arm_law.k == 2:
        out = {
            "ey0": out["ey[0]"],
            "ey1": out["ey[1]"],
            "ate": out["ate[1 vs 0]"],
            "log_rr": out["log_rr[1 vs 0]"],
            "par": out["par[0]"],
            "paf": out["paf[0]"],
        }
    return out


def control_bounds() -> dict[str, float]:
    """The scale of each derived ``noise_control``: the SD of the label's estimate.

    ``calibration_controls`` divides the bound by ``sqrt(CALIBRATION_N)``.  Every label sits at
    ``n = CALIBRATION_N``, so every ``noise_control`` adds one standard deviation of the
    group's own estimate.
    """
    out = {}
    for label in CONTROL_LABELS:
        (cell,) = (c for c in CELLS if c.name == f"{label}__corrected")
        match = _NAME.match(cell.estimand)
        assert match is not None
        reference = 1 if "vs 1]" in cell.estimand or cell.estimand.endswith("[1]") else 0
        out[label] = exact_sds(cell.truth, reference)[cell.estimand]
    return out


# ---------------------------------------------------------------------------------- the fit


def estimator(fit: Fit) -> CTMLE:
    """The subject of one configuration: a per-arm ``CTMLE`` with parametric learners."""
    if fit.outcome == "oracle":
        outcome_learner: Any = laws.OracleOutcome(
            "three_arm_active" if fit.k == 3 else "binary_active"
        )
    elif fit.outcome == "shipped":
        outcome_learner = ShippedOutcome(arm_columns=fit.k - 1)
    else:
        outcome_learner = ParametricOutcome(arm_columns=fit.k - 1)
    return CTMLE(
        strategy="oat",
        oat_design="per_arm",
        outcome_learner=outcome_learner,
        treatment_learner=ParametricMechanism(degree=3),
        cross_fit=fit.cross_fit,
        n_folds=FOLDS,
        repeats=1,
        simultaneous=fit.simultaneous,
        estimands=fit.estimands,
        reference=fit.reference,
        g_bounds=laws.G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def fit_cleverly(frame: pd.DataFrame, fit: Fit) -> Any:
    """Fit the study's subject on one sample.

    An admitted fit must report ``"influence_curve"`` for every estimate it reports.  A
    ``"stack_failure_plugin"`` estimate is a failure of the replicate: the study measures the
    stacked curve, and a fall back to the shipped curve would measure a different estimator.  For
    a band fit this is also the declared band set rule: a replicate whose band set holds an
    estimate with another status is lost.
    """
    roles = {"weights": "wt"} if fit.weighted else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            estimator(fit)
            .fit(frame, outcome="Y", treatment="A", covariates=list(fit.covariates), **roles)
            .single()
        )
    if fit.outcome == "parametric":
        for name, estimate in result.estimates.items():
            if estimate.inference != "influence_curve":
                raise DataError(
                    f"{name} reports the inference status {estimate.inference!r}, not the "
                    "stacked curve"
                )
    return result


def initials(result: Any, estimands: tuple[str, ...]) -> dict[str, float]:
    """The untargeted plug-in of each primary estimand.

    The arm codes sort as the labels do on both laws, so position ``j`` is arm ``j``.
    """
    data = result.data
    means = [float(np.mean(result.nuisance.outcome.arms[code])) for code in data.arm_codes]
    observed = float(np.mean(data.outcome))
    out: dict[str, float] = {}
    for name in estimands:
        match = _NAME.match(name)
        assert match is not None
        kind, arm = match["name"], match["arm"]
        if kind in ("par", "paf"):
            out[name] = observed - means[0] if kind == "par" else 1.0 - means[0] / observed
        elif kind == "ey":
            out[name] = means[int(arm[1:-1])]
        else:
            first, second = (int(part) for part in arm[1:-1].split(" vs "))
            high, low = means[first], means[second]
            out[name] = {
                "ate": high - low,
                "rr": high / low,
                "or": _odds(high) / _odds(low),
            }[kind]
    return out


# ------------------------------------------------------------------------- the primary phase

#: The primary scenario -> its fit.  ``three_arm_active`` reads ``reference="0"``, the old
#: study's setting, so its estimates are named ``ate[1 vs 0]`` and ``ate[2 vs 0]``.
PRIMARY_FITS: Mapping[str, Fit] = {
    "binary_active": Fit("primary", ("par", "paf")),
    "three_arm_active": Fit("primary", ("ey", "ate", "rr", "or"), k=3, reference="0"),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample and its truth from an explicit seed."""
    if scenario not in SCENARIOS:
        raise KeyError(scenario)
    frame, _ = laws.LAWS[scenario].sample(n, seed)
    return frame, {name: truth(scenario, name) for name in SCENARIOS[scenario]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truths = drawn
    result = fit_cleverly(frame, PRIMARY_FITS[scenario])
    return primary_rows(
        result=result,
        truth=truths,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=SCENARIOS[scenario],
        initials=initials(result, SCENARIOS[scenario]),
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for scenario in SCENARIOS for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])
