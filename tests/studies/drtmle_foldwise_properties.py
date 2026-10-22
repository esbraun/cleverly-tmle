r"""Property cells of the fold-wise DR-TMLE study (X83 plan section 7.3).

Every cell fits ``ate`` with ``cross_fit=True`` and, except the leak control, the shipped
``reduced_crossfit="foldwise"``.  Declared before any run:

=========================================  =========  =====  ========  ===============================  ================
cell                                       role       n      R         configuration                    planning (SD)
=========================================  =========  =====  ========  ===============================  ================
``fold_locality/ate__foldwise``            positive   1,000  4,400     ``treatment_correct``            0.140 +/- 0.022
``fold_locality/ate__pooled``              control    1,000  4,400     the same draw, ``"pooled"``      0.396 +/- 0.022
``double_robustness/ate__benkeser_order``  positive   3,000  1,600     ``update_order="benkeser"``      0.072 +/- 0.035 P
``double_robustness/ate__bivariate``       positive   3,000  4,000     ``reduction="bivariate"``        0.110 +/- 0.035 P
``double_robustness/ate__bivariate_oc``    positive   3,000  1,600     the same, ``outcome_correct``    0.030 +/- 0.050
``double_robustness/ate__k5``              positive   3,000  2,000     ``n_folds=5``                    0.084 +/- 0.034 P
``double_robustness/ate__spline_reduced``  positive   3,000  2,800     spline reduced learners          0.097 +/- 0.034 P
``double_robustness/ate__both_wrong``      control    3,000  400       ``both_wrong``                   12.6 (registered)
``double_robustness/multi_arm_drtmle__``   positive   2,000  3,600     the multi-arm law                0.069 +/- 0.050
``treatment_correct``
``double_robustness/<label>__stratified``  positive   2,000  3,200     the stratified paper law,        0.097, 0.092,
                                                                       ``outcome_correct``              0.035, 0.026
                                                                                                        +/- 0.035
=========================================  =========  =====  ========  ===============================  ================

The primary phase (:mod:`tests.studies.drtmle_foldwise`) is the tenth unit: three regimes at
n = 3,000, R = 6,000.  Its planning values are ``treatment_correct`` ``ate`` 0.099 +/- 0.025 P,
``ey0`` -0.044, ``ey1`` 0.091 (pilot ``tc3000``, 1,200 replications); ``outcome_correct`` 0.114,
-0.090, 0.063 +/- 0.035 (pilot ``oc3000``, 800 replications); ``both_correct`` 0.066, -0.097,
-0.003 +/- 0.035 (``canonical-drtmle`` pooled).  ``P`` marks a posterior of the pooled
``treatment_correct`` regime (``tau`` bound 0.038, pooled mean 0.093).  Every unit reaches its
share, so :data:`FALLBACK` is empty.  At these budgets the family-wise predictive pass is about
0.963.

The paper-law cells fit one regime of the ``canonical-drtmle`` law each.  Under
``treatment_correct`` the outcome GLM omits ``W1 W2`` and the mechanism is correct; that is the
regime where the fold leak shows (pilot ``leak1000``: pooled ``+0.396`` SD, fold-wise ``+0.140``
SD at n = 1,000, 2,000 replications).  Each ``treatment_correct`` variant changes one setting of
the primary fit (review F6, F10, F11).  ``ate__bivariate_oc`` fits the bivariate reduction under
``outcome_correct``, where the mechanism is wrong and equation (10)'s slab tilt of the single
bivariate reduced mechanism is load bearing.  ``ate__both_wrong`` is the double-robustness
control: both nuisances are misspecified, so its bias must fall outside the margin.  The
stratified cells read the ``outcome_correct`` configuration of
:mod:`tests.studies.canonical_stratified_drtmle` (marginal ATE and the three stratum ATEs of one
fit).  That regime exercises the treatment-side reduced regression, which the paper-law
``treatment_correct`` variants do not load.  The multi-arm cell carries a second
``treatment_correct`` law.

**Stratified ``treatment_correct`` is not gated.**  The sizing pilot fitted it fold-wise and
pooled on one draw (``pilot_b.csv``, 200 replications).  The fold-wise rows read ``+0.291``
(marginal) and ``+0.304`` (``v1``) SD, already outside the 0.25 SD margin, against the 0.12 SD
the review set as the gating line (:data:`STRATIFIED_TC_PILOT`).  The support row ``D03``
states the finding as a limit.

**The multi-arm cell reads the registered draws.**  Its stream root is the root of
``canonical-multi-arm-drtmle``'s ``double_robustness/treatment_correct`` cell, and the fit keeps
that cell's settings (its learners, five folds drawn by the estimator at ``random_state=0``,
``ate[medium vs high]``) with ``reduced_crossfit="foldwise"``.  Replication ``r < 600`` is
therefore that cell's registered replication ``r``, on the same rows and the same split, which
plan section 7.4 needs for the RM18 re-read.  Replications 600 and above continue the root's
stream.

**The leak control** (``fold_locality``, the shared necessity rule).  One draw per replication
is fitted twice, fold-wise and pooled, at n = 1,000.  The fold-wise row must keep its bias
inside the equivalence margin and the pooled row must establish its bias outside it.  The joint
clause is the paired displacement ``|mean(pooled) - mean(foldwise)| / sd(foldwise)``, which must
reach :data:`FOLD_LOCALITY_DISPLACEMENT`.  A study whose fold-wise path were wired as pooled
reads a displacement of exactly zero, so it fails.  The control does not separate the slab's
tilt (C2) from the leave-two-out designs (C1); plan section 7.3 leaves C2 to the unit mutation
tests.

**Why the leak control sits at n = 1,000.**  The leak is a second-order term, ``O(1 / n)`` in the
estimate, while the error SD is ``O(n ** -0.5)``.  The standardized leak therefore grows as n
shrinks, which moves the pooled control away from the margin.  At n = 1,500 the pooled row read
``+0.317`` SD (pilot 3, 2,000 replications), only 0.067 SD outside the margin, and no budget up
to :data:`CEILING` sized the unit.  At n = 1,000 it reads ``+0.396`` SD and the fold-wise row
``+0.140`` SD, so each side has about 0.11 to 0.15 SD of headroom.  The ratio 0.396 / 0.317 =
1.25 is close to ``sqrt(1.5) = 1.22``.  The leak pair is planned from pilot 4 alone; no pilot
at another n is pooled into it.

**Red-row reading, declared before the run.**  The policy is ``gated``.  A red positive row is a
defect suspect: diagnose it, fix it, and re-run under a fresh declaration.  A failing control is
a design failure: the study cannot see what the control measures, and it is redesigned before
any reading.  No replication is ever added after a verdict.

**Sizes: the arm-pooled joint predictive rule.**  :data:`PILOTS` holds the planning summary of
every gated row, with its source.  Each row's planning value is its standardized bias
``mean(error) / sd(error)``, with Monte Carlo SE ``1 / sqrt(R_pilot)``.  The rows one fit
estimates are correlated, with the pilot's per-replication error correlation.  Separate pilots
and rows planned from registered data are independent.  The five ``treatment_correct`` ATE arms
at n = 3,000 (:data:`REGIME`) are arms of one regime: their mean and between-arm SD ``tau`` are
estimated together, ``tau`` at its one-sided 80% Q-profile upper bound, and each arm's planning
value is its normal-normal posterior.  :func:`planning_draws` integrates over this planning
uncertainty with :data:`PLANNING_DRAWS` joint draws from :data:`PLANNING_SEED`.  At each draw
the study's own Monte Carlo error is drawn too, and a unit passes when every clause of its rows
passes: the bias clause (the 99% Student interval inside, or for a control outside, the
margin), the leak displacement, and, for the primary rows, the coverage floor and the SE-ratio
sanity band.  The union-model SE band (0.1 to 10) of the property rows does not enter: every
pilot SE ratio lies in 0.87 to 1.23.  A unit's budget is the smallest multiple of 400 at or
above its floor whose predictive pass reaches the unit share ``0.95 ** (1 / len(UNITS))``.
:func:`family_pass` reads the family-wise pass jointly at the declared budgets.

**Fallback, declared before the run.**  A unit that no budget up to :data:`CEILING` sizes is
listed in :data:`FALLBACK`.  Its remedy is a larger n for the failing positive arm, planned from
a pilot at that n.  The margin is never widened.  :func:`assert_sized` refuses a run while
:data:`FALLBACK` is not empty.  It is empty at the declared budgets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from scipy.stats import t as student

from cleverly.estimators import DRTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_stratified_drtmle as stratified
from tests.studies import drtmle_foldwise as study
from tests.studies import multi_arm_drtmle_properties as multi_arm
from tests.studies.drtmle_foldwise import STUDY, Fit
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyBatch,
    PropertyCell,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    Necessity,
    apply_shared_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

PAPER_N = 3_000
LEAK_N = 1_000
MULTI_ARM_N = 2_000
STRATIFIED_N = 2_000
#: The nuisance regime of the leak pair and of the ``treatment_correct`` variants.
SCENARIO = "treatment_correct"
#: The nuisance regime of the stratified cells.
STRATIFIED_SCENARIO = "outcome_correct"

# --------------------------------------------------------------------------------- the sizes
#: The equivalence margin, in empirical standard deviations.  Never widened.
MARGIN = STUDY.margins.standardized_bias
#: The two-sided 99% normal critical value of the coverage and SE-ratio intervals.
Z = float(norm.ppf(0.5 + STUDY.margins.confidence_level / 2))
#: The family-wise pass probability the budgets must reach when nothing is wrong.
STUDY_TARGET = 0.95
#: The budget grid: multiples of 400, from a floor of 400 (1,200 for the primary phase, which
#: the brief asks to keep at or above 1,000) to a ceiling past which the fallback applies.
STEP = 400
PROPERTY_FLOOR = 400
PRIMARY_FLOOR = 1_200
CEILING = 8_000
#: The joint planning draws and their seed.  Fixed, so every number the rule returns is exact.
PLANNING_DRAWS = 20_000
PLANNING_SEED = 2026102283
#: The one-sided level of the between-arm SD's upper bound.
TAU_LEVEL = 0.80

_PILOT_A = "x83 sizing pilot, reviews/x83/pilot2/pilot_a.csv"
_PILOT_B = "x83 sizing pilot, reviews/x83/pilot2/pilot_b.csv"
#: The extension pilot of the three units pilot 2 could not size, same code and columns.
_PILOT_C = "x83 sizing pilot, reviews/x83/pilot3/pilot_c"
#: The pilot of the leak pair at n = 1,000 and of the primary ``outcome_correct`` rows at
#: n = 3,000, same code and columns.
_PILOT_D = "x83 sizing pilot, reviews/x83/pilot4/pilot_d"


@dataclass(frozen=True)
class Pilot:
    """The planning summary of the rows one fit estimates on one draw.

    Parameters
    ----------
    source : str
        Where the summary comes from.  ``reviews/x83`` is in the plans repository, and its
        pilots draw from ``stream_seed(STUDY, "x83_sizing_pilot", cell, r)``, a label no study
        stream uses.
    replicates : int
        The replications the summary reads.
    rows : tuple of str
        The rows, each an estimand or an arm.
    bias : tuple of float
        Each row's standardized bias, ``mean(error) / sd(error)``.
    correlation : tuple of tuple of float
        The per-replication correlation of the rows' errors.
    controls : tuple of str
        The rows whose bias must fall outside the margin.
    se_ratio : tuple of float
        Each row's mean reported SE over its error SD; given for the primary rows only.
    coverage : tuple of float
        Each row's 95% Wald coverage; given for the primary rows only.
    """

    source: str
    replicates: int
    rows: tuple[str, ...]
    bias: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    controls: tuple[str, ...] = ()
    se_ratio: tuple[float, ...] = ()
    coverage: tuple[float, ...] = ()


_ONE = ((1.0,),)
_ATE_EY = ("ate", "ey0", "ey1")

#: The planning summary of every gated row.  The primary ``both_correct`` rows and the both-wrong control read the committed ``canonical-drtmle``
#: pooled rows at n = 3,000; the X83 probe P1 read fold-wise ``|bias|`` below pooled ``|bias|`` in every regime
#: and estimand, so a pooled value is an upper planning value for its fold-wise row.
PILOTS: Mapping[str, Pilot] = {
    "primary/treatment_correct": Pilot(
        f"{_PILOT_A} r < 400 and {_PILOT_C}_tc.csv r 400 to 1,199, cell tc3000",
        1_200,
        _ATE_EY,
        (0.1032, -0.0469, 0.0939),
        ((1.0, -0.6727, 0.6986), (-0.6727, 1.0, 0.0596), (0.6986, 0.0596, 1.0)),
        se_ratio=(0.9815, 1.0123, 0.9924),
        coverage=(0.9392, 0.9442, 0.9483),
    ),
    "primary/outcome_correct": Pilot(
        f"{_PILOT_D}_oc3000.csv r < 800, cell oc3000",
        800,
        _ATE_EY,
        (0.1138, -0.0901, 0.0628),
        ((1.0, -0.6567, 0.6829), (-0.6567, 1.0, 0.1025), (0.6829, 0.1025, 1.0)),
        se_ratio=(0.9712, 0.9726, 0.9692),
        coverage=(0.9462, 0.9450, 0.9412),
    ),
    "primary/both_correct": Pilot(
        "canonical-drtmle summary.csv and replicates.parquet, pooled, n = 3,000",
        800,
        _ATE_EY,
        (0.0659, -0.0969, -0.0035),
        ((1.0, -0.6467, 0.7089), (-0.6467, 1.0, 0.0795), (0.7089, 0.0795, 1.0)),
        se_ratio=(0.9917, 1.0351, 0.9765),
        coverage=(0.94375, 0.9575, 0.9425),
    ),
    "fold_locality": Pilot(
        f"{_PILOT_D}_leak1000.csv r < 2,000, cell leak1000",
        2_000,
        ("foldwise", "pooled"),
        (0.1403, 0.3956),
        ((1.0, 0.8544), (0.8544, 1.0)),
        controls=("pooled",),
    ),
    "benkeser_order": Pilot(f"{_PILOT_A}, cell benkeser_order", 400, ("ate",), (0.0364,), _ONE),
    "bivariate": Pilot(f"{_PILOT_A}, cell bivariate", 400, ("ate",), (0.1427,), _ONE),
    "bivariate_oc": Pilot(f"{_PILOT_A}, cell bivariate_oc", 400, ("ate",), (0.0298,), _ONE),
    "k5": Pilot(f"{_PILOT_A}, cell k5", 400, ("ate",), (0.0702,), _ONE),
    "spline_reduced": Pilot(
        f"{_PILOT_A}, cell adaptive_reduced (the spline cell's former name)",
        400,
        ("ate",),
        (0.1042,),
        _ONE,
    ),
    "multi_arm": Pilot(f"{_PILOT_A}, cell multi_arm", 400, ("ate",), (0.0691,), _ONE),
    "stratified": Pilot(
        f"{_PILOT_B} r < 200 and {_PILOT_C}_strat.csv r 200 to 799, cell strat_oc",
        800,
        stratified.ATE_LABELS,
        (0.0965, 0.0916, 0.0354, 0.0261),
        (
            (1.0, 0.7320, 0.5229, 0.4722),
            (0.7320, 1.0, 0.0211, 0.0575),
            (0.5229, 0.0211, 1.0, -0.0398),
            (0.4722, 0.0575, -0.0398, 1.0),
        ),
    ),
    "both_wrong": Pilot(
        "canonical-drtmle properties.csv, double_robust_contraction/both_wrong_n3000, pooled",
        800,
        ("ate",),
        (12.6089,),
        _ONE,
        controls=("ate",),
    ),
}

#: The arms of the ``treatment_correct`` regime at n = 3,000: the primary ATE and the four
#: one-setting variants.  The first row of each pilot is the arm.
REGIME = ("primary/treatment_correct", "benkeser_order", "bivariate", "k5", "spline_reduced")

#: The leak pilot's ``sd(pooled error) / sd(foldwise error)``, which turns the two rows'
#: standardized biases into the paired displacement: ``|ratio * b_pooled - b_foldwise|``.  The
#: pilot's displacement is ``0.266``, with per-replication SD ``0.547``.
LEAK_SPREAD_RATIO = 1.0260

#: The declared joint threshold of the leak control.  The pilot's ``0.266 +/- 0.012`` sits
#: thirteen of its standard errors above it; a construction wired as pooled reads exactly zero.
FOLD_LOCALITY_DISPLACEMENT = 0.10

#: The stratified ``treatment_correct`` rows of ``pilot_b.csv`` (200 replications, one draw
#: fitted both ways): ``label -> (foldwise, pooled)`` standardized bias.  The review gates the
#: regime only if every fold-wise row plans at or below :data:`STRATIFIED_TC_GATE`.
STRATIFIED_TC_PILOT: Mapping[str, tuple[float, float]] = {
    "marginal_ate": (0.2905, 0.5823),
    "v0_ate": (0.0391, 0.2913),
    "v1_ate": (0.3035, 0.4242),
    "v2_ate": (0.1837, 0.3313),
}
STRATIFIED_TC_GATE = 0.12

#: The units of the rule: each fit group, the pilots of its rows, and its budget floor.
UNITS: Mapping[str, tuple[tuple[str, ...], int]] = {
    "primary": (
        ("primary/treatment_correct", "primary/outcome_correct", "primary/both_correct"),
        PRIMARY_FLOOR,
    ),
    **{
        name: ((name,), PROPERTY_FLOOR)
        for name in (
            "fold_locality",
            "benkeser_order",
            "bivariate",
            "bivariate_oc",
            "k5",
            "spline_reduced",
            "both_wrong",
            "multi_arm",
            "stratified",
        )
    },
}
UNIT_TARGET = STUDY_TARGET ** (1.0 / len(UNITS))


def _factor(correlation: tuple[tuple[float, ...], ...]) -> np.ndarray:
    """A square root of a correlation matrix that may be singular (``ate = ey1 - ey0``)."""
    values, vectors = np.linalg.eigh(np.asarray(correlation, dtype=float))
    return np.asarray(vectors * np.sqrt(np.clip(values, 0.0, None)))


def regime_tau() -> tuple[float, float, float]:
    """The regime's between-arm SD at its :data:`TAU_LEVEL` upper bound, and the pooled mean.

    The bound is the one-sided Q-profile bound: the ``tau`` at which the generalized ``Q``
    statistic equals the ``1 - TAU_LEVEL`` quantile of chi-square on ``k - 1`` degrees of
    freedom, or zero when ``Q`` at zero already lies below it.

    Returns
    -------
    tuple of float
        ``(tau, pooled mean, SE of the pooled mean)``.
    """
    y = np.array([PILOTS[name].bias[0] for name in REGIME])
    v = np.array([1.0 / PILOTS[name].replicates for name in REGIME])

    def q(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = float(np.sum(w * y) / np.sum(w))
        return float(np.sum(w * (y - mu) ** 2))

    critical = float(chi2.ppf(1.0 - TAU_LEVEL, len(y) - 1))
    t2 = 0.0 if q(0.0) <= critical else float(brentq(lambda t: q(t) - critical, 0.0, 1.0))
    w = 1.0 / (v + t2)
    return math.sqrt(t2), float(np.sum(w * y) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))


@dataclass(frozen=True)
class _Draws:
    bias: Mapping[str, np.ndarray]
    se_ratio: Mapping[str, np.ndarray]
    coverage: Mapping[str, np.ndarray]
    noise: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


@cache
def planning_draws() -> _Draws:
    """The joint planning draws and the study's own Monte Carlo noise, from one seed.

    Each pilot's rows are drawn jointly normal around their summary with covariance
    ``correlation / replicates``.  The :data:`REGIME` arms are then replaced by their
    normal-normal posterior at the bounded ``tau``, with the pooled mean drawn once per planning
    draw, so the five arms share its error.  The primary ``treatment_correct`` ``ey0`` and
    ``ey1`` follow their ATE by the pilot's regression.  The SE ratio and coverage of a primary
    row are drawn around the pilot's value with SEs ``ratio / sqrt(2 R)`` and
    ``sqrt(c (1 - c) / R)``.

    Returns
    -------
    _Draws
        The planning values and standard normal noise for the study's realization.
    """
    rng = np.random.default_rng(PLANNING_SEED)
    bias: dict[str, np.ndarray] = {}
    for name, pilot in PILOTS.items():
        shocks = rng.standard_normal((PLANNING_DRAWS, len(pilot.rows)))
        bias[name] = np.asarray(pilot.bias) + shocks @ _factor(pilot.correlation).T / math.sqrt(
            pilot.replicates
        )
    tau, mean, spread = regime_tau()
    mu = mean + spread * rng.standard_normal(PLANNING_DRAWS)
    for name in REGIME:
        pilot = PILOTS[name]
        v = 1.0 / pilot.replicates
        shrink = tau**2 / (v + tau**2)
        arm = (
            mu
            + shrink * (pilot.bias[0] - mu)
            + math.sqrt(v * shrink) * rng.standard_normal(PLANNING_DRAWS)
        )
        moved = arm - bias[name][:, 0]
        others = np.asarray(pilot.correlation, dtype=float)[1:, 0]
        bias[name][:, 1:] += np.outer(moved, others)
        bias[name][:, 0] = arm
    se_ratio: dict[str, np.ndarray] = {}
    coverage: dict[str, np.ndarray] = {}
    for name, pilot in PILOTS.items():
        if not pilot.se_ratio:
            continue
        ratio = np.asarray(pilot.se_ratio)
        covered = np.asarray(pilot.coverage)
        shape = (PLANNING_DRAWS, len(pilot.rows))
        se_ratio[name] = ratio + rng.standard_normal(shape) * ratio / math.sqrt(
            2 * pilot.replicates
        )
        coverage[name] = np.clip(
            covered
            + rng.standard_normal(shape) * np.sqrt(covered * (1 - covered) / pilot.replicates),
            0.0,
            1.0,
        )
    noise = {
        name: tuple(rng.standard_normal((PLANNING_DRAWS, len(pilot.rows))) for _ in range(3))
        for name, pilot in PILOTS.items()
    }
    return _Draws(bias, se_ratio, coverage, noise)  # type: ignore[arg-type]


#: The clauses the rule reads.  ``se_ratio`` and ``coverage`` bind the primary rows only.
CLAUSES = ("bias", "displacement", "se_ratio", "coverage")


def pilot_passes(name: str, replicates: int, clauses: tuple[str, ...] = CLAUSES) -> np.ndarray:
    """Whether one pilot's rows all pass at ``replicates``, at each planning draw.

    Parameters
    ----------
    name : str
        A key of :data:`PILOTS`.
    replicates : int
        The budget.
    clauses : tuple of str
        The clauses read, a subset of :data:`CLAUSES`; a test drops one to witness it.

    Returns
    -------
    numpy.ndarray
        One boolean per planning draw.
    """
    pilot = PILOTS[name]
    draws = planning_draws()
    bias_noise, ratio_noise, coverage_noise = draws.noise[name]
    root = math.sqrt(replicates)
    half = float(student.ppf(0.5 + STUDY.margins.confidence_level / 2, replicates - 1)) / root
    estimate = draws.bias[name] + bias_noise @ _factor(pilot.correlation).T / root
    passed = np.ones(PLANNING_DRAWS, dtype=bool)
    for index, row in enumerate(pilot.rows):
        if "bias" not in clauses:
            break
        size = np.abs(estimate[:, index])
        passed &= size - half > MARGIN if row in pilot.controls else size + half <= MARGIN
    if name == "fold_locality" and "displacement" in clauses:
        moved = np.abs(LEAK_SPREAD_RATIO * estimate[:, 1] - estimate[:, 0])
        passed &= moved >= FOLD_LOCALITY_DISPLACEMENT
    if pilot.se_ratio and "se_ratio" in clauses:
        ratio = draws.se_ratio[name]
        read = ratio + ratio_noise * ratio / math.sqrt(2 * replicates)
        width = Z * read / math.sqrt(2 * replicates)
        low, high = STUDY.margins.se_ratio_sanity
        passed &= ((read - width >= low) & (read + width <= high)).all(axis=1)
    if pilot.coverage and "coverage" in clauses:
        covered = draws.coverage[name]
        read = np.clip(
            covered + coverage_noise * np.sqrt(covered * (1 - covered) / replicates), 0.0, 1.0
        )
        floor = read - Z * np.sqrt(read * (1 - read) / replicates)
        passed &= (floor >= STUDY.margins.coverage_floor).all(axis=1)
    return passed


def unit_passes(unit: str, replicates: int, clauses: tuple[str, ...] = CLAUSES) -> np.ndarray:
    """Whether every row of a unit passes at ``replicates``, at each planning draw."""
    names, _ = UNITS[unit]
    passed = np.ones(PLANNING_DRAWS, dtype=bool)
    for name in names:
        passed &= pilot_passes(name, replicates, clauses)
    return passed


def unit_pass(unit: str, replicates: int, clauses: tuple[str, ...] = CLAUSES) -> float:
    """A unit's predictive pass probability at ``replicates``."""
    return float(unit_passes(unit, replicates, clauses).mean())


@cache
def budget(unit: str) -> tuple[int, float, bool]:
    """``(R, predictive pass, sized)``: the smallest grid budget reaching :data:`UNIT_TARGET`.

    A unit that no budget up to :data:`CEILING` sizes returns the ceiling and ``False``.
    """
    _, floor = UNITS[unit]
    for replicates in range(floor, CEILING + 1, STEP):
        value = unit_pass(unit, replicates)
        if value >= UNIT_TARGET:
            return replicates, value, True
    return CEILING, unit_pass(unit, CEILING), False


def family_pass(budgets: Mapping[str, int] | None = None) -> float:
    """The family-wise predictive pass probability, read jointly over the planning draws."""
    chosen = BUDGETS if budgets is None else budgets
    passed = np.ones(PLANNING_DRAWS, dtype=bool)
    for unit in UNITS:
        passed &= unit_passes(unit, chosen[unit])
    return float(passed.mean())


#: The declared budgets, each the output of :func:`budget`.
BUDGETS: Mapping[str, int] = {
    "primary": 6_000,
    "fold_locality": 4_400,
    "benkeser_order": 1_600,
    "bivariate": 4_000,
    "bivariate_oc": 1_600,
    "k5": 2_000,
    "spline_reduced": 2_800,
    "both_wrong": 400,
    "multi_arm": 3_600,
    "stratified": 3_200,
}

#: The units no budget up to :data:`CEILING` sizes, each the output of :func:`budget`.
FALLBACK: tuple[str, ...] = ()


def assert_sized() -> None:
    """Refuse a run while a unit waits on its fallback.

    Raises
    ------
    RuntimeError
        When :data:`FALLBACK` is not empty.
    """
    if FALLBACK:
        raise RuntimeError(
            f"drtmle-foldwise units {', '.join(FALLBACK)} reach no family-wise share by "
            f"R = {CEILING}; their declared fallback (a larger n, planned from a pilot at "
            "that n) is open"
        )


# ------------------------------------------------------------------------------ the cells
LEAK = Necessity(
    family="fold_locality",
    labels=("ate",),
    arms=("foldwise", "pooled"),
    column="fold_locality_displacement",
    threshold=FOLD_LOCALITY_DISPLACEMENT,
)


@dataclass(frozen=True)
class Group:
    """One paper-law or stratified fit group: one sample per replication, the fits made on it.

    Parameters
    ----------
    name : str
        The group's name; it labels the sample stream and names its unit in :data:`UNITS`.
    law : str
        ``"paper"`` or ``"stratified"``.
    scenario : str
        The nuisance regime every fit of the group uses.
    n : int
        The sample size.
    fits : tuple of tuple
        ``(family, cell, role, Fit)`` for each fit made on the sample.
    """

    name: str
    law: str
    scenario: str
    n: int
    fits: tuple[tuple[str, str, str, Fit], ...]

    @property
    def replicates(self) -> int:
        """The group's budget."""
        return BUDGETS[self.name]

    @property
    def n_folds(self) -> int:
        """The fold count of the sample's vector."""
        return self.fits[0][3].n_folds


def _variant(name: str, scenario: str, fit: Fit, role: str = "positive") -> Group:
    return Group(
        name, "paper", scenario, PAPER_N, (("double_robustness", f"ate__{name}", role, fit),)
    )


GROUPS: Mapping[str, Group] = {
    "fold_locality": Group(
        "fold_locality",
        "paper",
        SCENARIO,
        LEAK_N,
        (
            ("fold_locality", "ate__foldwise", "positive", Fit()),
            ("fold_locality", "ate__pooled", "control", Fit(reduced_crossfit=study.POOLED)),
        ),
    ),
    "benkeser_order": _variant("benkeser_order", SCENARIO, Fit(update_order="benkeser")),
    "bivariate": _variant("bivariate", SCENARIO, Fit(reduction="bivariate")),
    "bivariate_oc": _variant("bivariate_oc", "outcome_correct", Fit(reduction="bivariate")),
    "k5": _variant("k5", SCENARIO, Fit(n_folds=5)),
    "spline_reduced": _variant("spline_reduced", SCENARIO, Fit(reduced="spline")),
    "both_wrong": _variant("both_wrong", "both_wrong", Fit(), role="control"),
    "stratified": Group(
        "stratified",
        "stratified",
        STRATIFIED_SCENARIO,
        STRATIFIED_N,
        tuple(
            ("double_robustness", f"{label}__stratified", "positive", Fit())
            for label in stratified.ATE_LABELS
        ),
    ),
}

#: The registered multi-arm cell whose stream and settings the multi-arm cell reuses.
MULTI_ARM_REGISTERED = ("double_robustness", SCENARIO)
MULTI_ARM_CELL = "multi_arm_drtmle__treatment_correct"


def _registered_multi_arm() -> PropertyCell:
    (cell,) = [
        cell for cell in multi_arm.cells() if (cell.property, cell.cell) == MULTI_ARM_REGISTERED
    ]
    return cell


def multi_arm_cell() -> PropertyCell:
    """The multi-arm cell: the registered cell's law, nuisances, size and root, renamed."""
    registered = _registered_multi_arm()
    return PropertyCell(
        "double_robustness",
        MULTI_ARM_CELL,
        registered.dgp,
        registered.outcome_learner,
        registered.treatment_learner,
        registered.n,
        BUDGETS["multi_arm"],
        registered.seed,
        role="positive",
        estimand=registered.estimand,
    )


def multi_arm_settings(cell: PropertyCell) -> dict[str, Any]:
    """The registered cell's estimator arguments, with the fold-wise reduced regressions."""
    return {**multi_arm._settings(cell), "reduced_crossfit": study.FOLDWISE}


def _multi_arm_estimator(cell: PropertyCell) -> Any:
    return lambda: DRTMLE(**multi_arm_settings(cell))


def multi_arm_batch() -> PropertyBatch:
    """The multi-arm loop, through the shared cell runner."""
    return PropertyBatch("multi_arm", (multi_arm_cell(),), _multi_arm_estimator)


# ------------------------------------------------------------------------------- the fits
def stratified_fit(frame: pd.DataFrame, fit: Fit = Fit()) -> Any:
    """The ``canonical-stratified-drtmle`` fit of the ``outcome_correct`` configuration.

    The call is that study's ``fit_cleverly`` with ``reduced_crossfit`` replaced.
    """
    return stratified.fit_cleverly(
        frame, STRATIFIED_SCENARIO, reduced_crossfit=fit.reduced_crossfit
    )


def sample_seed(group: str, replicate: int) -> int:
    """The sample seed of one replication of a paper-law or stratified group."""
    return stream_seed(STUDY, "property_sample", group, replicate)


_Key = tuple[tuple[str, int, int]]
_Drawn = tuple[pd.DataFrame, dict[str, float]]


def _draw(key: _Key) -> _Drawn:
    """The sample a key fits, a function of the key alone."""
    name, replicate, _ = key[0]
    group = GROUPS[name]
    seed = sample_seed(name, replicate)
    if group.law == "stratified":
        return stratified.draw_from_seed(stratified.SCENARIO, group.n, seed)
    return study.draw_paper(group.scenario, group.n, seed, group.n_folds)


def _cells_of(key: _Key) -> list[str]:
    return [f"{family}/{cell}" for family, cell, _, _ in GROUPS[key[0][0]].fits]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[0][1]


def _fit_key(key: _Key, drawn: _Drawn) -> list[dict[str, Any]]:
    name, replicate, requested = key[0]
    group = GROUPS[name]
    frame, truths = drawn
    rows: list[dict[str, Any]] = []
    if group.law == "stratified":
        result = stratified_fit(frame)
        for (family, cell, role, _), label in zip(group.fits, stratified.ATE_LABELS, strict=True):
            target = stratified.ate_name(label)
            rows.append(
                replicate_row(
                    property_name=family,
                    cell=cell,
                    role=role,
                    replicate=replicate,
                    n=group.n,
                    requested=requested,
                    truth=float(truths[target]),
                    estimate=result[target],
                    alpha=STUDY.margins.alpha,
                )
            )
        return rows
    for family, cell, role, fit in group.fits:
        result = study.fit_cleverly(frame, group.scenario, fit)
        rows.append(
            replicate_row(
                property_name=family,
                cell=cell,
                role=role,
                replicate=replicate,
                n=group.n,
                requested=requested,
                truth=float(truths["ate"]),
                estimate=result.estimates["ate"],
                alpha=STUDY.margins.alpha,
            )
        )
    return rows


def payloads(budget: int | None = None) -> list[_Key]:
    """Every paper-law and stratified replication; ``budget`` caps each group for a smoke run."""
    out: list[_Key] = []
    for name, group in GROUPS.items():
        count = group.replicates if budget is None else min(budget, group.replicates)
        out += [((name, replicate, count),) for replicate in range(count)]
    return out


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: before the study's first run.  Up to ``floor(0.01 R)`` lost replicates are judged by the worst
#: case of each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each loop for a smoke run.

    A declared run (``budget=None``) is refused while :data:`FALLBACK` is not empty.
    """
    if budget is None:
        assert_sized()
    outcomes = fit_properties(
        payloads(budget),
        _fit_key,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    paper = pd.DataFrame([row for outcome in outcomes for row in outcome])
    batch = multi_arm_batch()
    if budget is not None:
        (cell,) = batch.cells
        capped = PropertyCell(
            cell.property,
            cell.cell,
            cell.dgp,
            cell.outcome_learner,
            cell.treatment_learner,
            cell.n,
            min(budget, cell.replicates),
            cell.seed,
            role=cell.role,
            estimand=cell.estimand,
        )
        batch = PropertyBatch(batch.name, (capped,), batch.estimator)
    arms = batch.run(n_jobs=n_jobs)
    rows = pd.concat([paper, arms], ignore_index=True)
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared double-robustness rule, then the leak control's necessity rule."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=(LEAK.column,), rate_labels=()
    )
    necessity_verdicts(summary, rows, LEAK)
    return finish(summary, rates)


#: The failure accounting's declared rules.  ``double_robustness`` answers to the shared rule.
PROPERTY_RULES = (property_verdicts.necessity_rule(LEAK),)


def property_cells() -> dict[str, tuple[str, ...]]:
    """The ``family -> cells`` map the record declares, built from the groups."""
    out: dict[str, list[str]] = {}
    for group in GROUPS.values():
        for family, cell, _, _ in group.fits:
            out.setdefault(family, []).append(cell)
    out["double_robustness"].append(MULTI_ARM_CELL)
    return {family: tuple(cells) for family, cells in out.items()}
