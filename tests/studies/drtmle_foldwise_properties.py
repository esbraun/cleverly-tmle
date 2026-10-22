r"""Property cells of the fold-wise DR-TMLE study (X83 plan section 7.3).

Every cell fits ``ate`` with ``cross_fit=True`` and, except the leak control, the shipped
``reduced_crossfit="foldwise"``.  Declared before any run:

========================================  =========  =====  ======  ===============================
cell                                      role       n      R       configuration
========================================  =========  =====  ======  ===============================
``fold_locality/ate__foldwise``           positive   2,000  4,400    paper law, ``treatment_correct``
``fold_locality/ate__pooled``             control    2,000  4,400    the same draw, ``"pooled"``
``double_robustness/ate__benkeser_order``  positive   3,000  2,000   ``update_order="benkeser"``
``double_robustness/ate__bivariate``       positive   3,000  2,000   ``reduction="bivariate"``
``double_robustness/ate__k5``              positive   3,000  2,000   ``n_folds=5``
``double_robustness/ate__adaptive_reduced`` positive  3,000  2,000   spline reduced learners
``double_robustness/multi_arm_drtmle__``  positive   2,000  3,200   the multi-arm law
``treatment_correct``
``double_robustness/<label>__stratified`` positive   2,000  1,600   the stratified paper law,
                                                                     ``outcome_correct``
========================================  =========  =====  ======  ===============================

The four paper-law variants and the leak pair fit the ``treatment_correct`` regime of the
``canonical-drtmle`` law: the outcome GLM omits ``W1 W2`` and the mechanism is correct.  That is
the regime where the pilot P1 measured the fold leak (pooled ``+0.339`` SD, fold-wise ``+0.134``
SD, paired ``+0.227 +/- 0.025`` SD at n = 2,000).  Each variant changes one setting of the
primary fit (review F6, F10, F11).  The stratified cells read the ``outcome_correct``
configuration of :mod:`tests.studies.canonical_stratified_drtmle` (marginal ATE and the three
stratum ATEs of one fit).  That regime exercises the treatment-side reduced regression, the
channel the paper-law variants do not load; its registered pooled biases (0.04 to 0.11 SD at
n = 2,000) leave room under the margin, where the ``treatment_correct`` marginal row (0.60 SD
pooled) would not.  The multi-arm cell carries a second ``treatment_correct`` law.

**The multi-arm cell reads the registered draws.**  Its stream root is the root of
``canonical-multi-arm-drtmle``'s ``double_robustness/treatment_correct`` cell, and the fit keeps
that cell's settings (its learners, five folds drawn by the estimator at ``random_state=0``,
``ate[medium vs high]``) with ``reduced_crossfit="foldwise"``.  Replication ``r < 600`` is
therefore that cell's registered replication ``r``, on the same rows and the same split, which
plan section 7.4 needs for the RM18 re-read.  Replications 600 and above continue the root's
stream.

**The leak control** (``fold_locality``, the shared necessity rule).  One draw per replication
is fitted twice, fold-wise and pooled.  The fold-wise row must keep its bias inside the
equivalence margin and the pooled row must establish its bias outside it.  The joint clause is
the paired displacement ``|mean(pooled) - mean(foldwise)| / sd(foldwise)``, which must reach
:data:`FOLD_LOCALITY_DISPLACEMENT`.  A study whose fold-wise path were wired as pooled reads a
displacement of exactly zero, so it fails.  The control does not separate the slab's tilt (C2)
from the leave-two-out designs (C1); plan section 7.3 leaves C2 to the unit mutation tests.

**Red-row reading, declared before the run.**  The policy is ``gated``.  A red positive row is a
defect suspect: diagnose it, fix it, and re-run under a fresh declaration.  A leak control that
fails is a design failure: the study cannot see the leak, and it is redesigned before any
reading.  No replication is ever added after a verdict.

**Sizes.**  :func:`budget` gives every budget in :data:`BUDGETS`, and
``tests/unit/test_drtmle_foldwise_study_design.py`` pins the two equal.  The rule is the point
rule at conservative planning values: the smallest multiple of 400, at or above the unit's floor,
whose pass probability at the :data:`PLANNING` points reaches ``0.95 ** (1 / 8)``.  The
study-wise point probability is about 0.97.  The predictive probability, which averages over the
planning values' Monte Carlo error (:func:`study_pass`), is about 0.28 at these budgets and does
not reach 0.95 at any budget: the planning SEs (0.03 to 0.05 SD) are a fifth of the 0.25 SD
margin.  A declared-size pilot of the paper-law rows should precede the run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

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
LEAK_N = 2_000
MULTI_ARM_N = 2_000
STRATIFIED_N = 2_000
#: The nuisance regime of every paper-law property cell.
SCENARIO = "treatment_correct"
#: The nuisance regime of the stratified cells.
STRATIFIED_SCENARIO = "outcome_correct"

# --------------------------------------------------------------------------------- the sizes
#: The two-sided 99% critical value of the bias interval.
Z = float(norm.ppf(0.995))
#: The equivalence margin, in empirical standard deviations.
MARGIN = STUDY.margins.standardized_bias
#: The study-wise pass probability the budgets must reach when nothing is wrong.
STUDY_TARGET = 0.95
#: The budget grid: multiples of 400, from a floor of 400 (1,200 for the primary phase, which
#: the brief asks to keep at or above 1,000) to a ceiling past which the rule stops.
STEP = 400
PROPERTY_FLOOR = 400
PRIMARY_FLOOR = 1_200
CEILING = 8_000

#: The planning value of each gated bias row: ``(standardized bias, its Monte Carlo SE,
#: source)``.  The verdict of a row whose bias clause binds is the 99% Student interval of the
#: bias inside +/- 0.25 SD; the SE-ratio, coverage and union-band clauses do not bind at these
#: budgets on any pilot below (SE ratios 0.93 to 1.06).
#:
#: ``P1`` is the X83 probe (``reviews/x83/p1`` in the plans repository): 400 fold-wise fits at
#: n = 2,000 on ``canonical-drtmle`` draws, and 400 pooled fits on the same draws.  A row at
#: n = 3,000 is planned at its n = 2,000 value, unscaled, which is conservative for a
#: second-order remainder.  P1 read fold-wise ``|bias|`` below pooled ``|bias|`` in every regime
#: and estimand, so a registered pooled value is an upper planning value for its fold-wise row:
#: ``canonical-drtmle`` (n = 3,000, R = 800), ``canonical-multi-arm-drtmle``
#: (``double_robustness/treatment_correct``, n = 2,000, R = 600) and
#: ``canonical-stratified-drtmle`` (``outcome_correct``, n = 2,000).  The paper-law variants are
#: planned at P1's fold-wise value.  The pooled leak row combines P1 (0.339 +/- 0.05) and the
#: committed ``canonical-drtmle`` ``double_robustness/treatment_correct`` row at n = 1,500
#: (0.326 +/- 0.035) by inverse variance.
PLANNING: Mapping[str, tuple[float, float, str]] = {
    "primary/treatment_correct/ate": (0.134, 0.050, "P1"),
    "primary/treatment_correct/ey0": (-0.099, 0.050, "P1"),
    "primary/treatment_correct/ey1": (0.085, 0.050, "P1"),
    "primary/outcome_correct/ate": (0.027, 0.050, "P1"),
    "primary/outcome_correct/ey0": (-0.018, 0.050, "P1"),
    "primary/outcome_correct/ey1": (0.017, 0.050, "P1"),
    "primary/both_correct/ate": (0.066, 0.035, "canonical-drtmle pooled"),
    "primary/both_correct/ey0": (-0.097, 0.035, "canonical-drtmle pooled"),
    "primary/both_correct/ey1": (-0.004, 0.035, "canonical-drtmle pooled"),
    "fold_locality/ate__foldwise": (0.134, 0.050, "P1"),
    "fold_locality/ate__pooled": (0.330, 0.029, "P1 + canonical-drtmle property row"),
    "benkeser_order/ate": (0.134, 0.050, "P1"),
    "bivariate/ate": (0.134, 0.050, "P1"),
    "k5/ate": (0.134, 0.050, "P1"),
    "adaptive_reduced/ate": (0.134, 0.050, "P1"),
    "multi_arm/ate": (0.160, 0.041, "canonical-multi-arm-drtmle pooled"),
    "stratified/marginal_ate": (0.110, 0.035, "canonical-stratified-drtmle pooled"),
    "stratified/v0_ate": (0.086, 0.035, "canonical-stratified-drtmle pooled"),
    "stratified/v1_ate": (0.056, 0.035, "canonical-stratified-drtmle pooled"),
    "stratified/v2_ate": (0.041, 0.035, "canonical-stratified-drtmle pooled"),
}

#: The paired leak displacement ``|mean(pooled) - mean(foldwise)| / sd(foldwise)``: its P1
#: value and the per-replication SD of the paired difference in the same units (``0.025 *
#: sqrt(400)``).
LEAK_PAIRED = (0.227, 0.50)

#: The declared joint threshold of the leak control.  Less than half of P1's ``0.227``, and
#: ``0.227 - 0.10`` is five P1 standard errors, so the control resolves the leak P1 measured;
#: a construction wired as pooled reads exactly zero.
FOLD_LOCALITY_DISPLACEMENT = 0.10


def bias_pass(bias: Any, replicates: int) -> np.ndarray:
    """The probability that a row with true standardized bias ``bias`` passes at ``replicates``.

    The 99% interval of the standardized bias is about ``b_hat +/- Z / sqrt(R)`` with
    ``b_hat ~ N(b, 1 / R)``; the row passes when it lies inside ``+/- MARGIN``.

    Parameters
    ----------
    bias : array_like
        True standardized biases.
    replicates : int
        The budget.

    Returns
    -------
    numpy.ndarray
        The pass probabilities.
    """
    b = np.asarray(bias, dtype=float)
    root = math.sqrt(replicates)
    inside = norm.cdf(root * (MARGIN - b) - Z) - norm.cdf(-root * (MARGIN + b) + Z)
    return np.clip(inside, 0.0, 1.0)


def beyond_pass(bias: Any, replicates: int) -> np.ndarray:
    """The probability that a control row establishes its bias outside ``+/- MARGIN``."""
    b = np.abs(np.asarray(bias, dtype=float))
    return np.asarray(norm.cdf(math.sqrt(replicates) * (b - MARGIN) - Z))


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def predictive(row: str, replicates: int) -> float:
    """A row's pass probability, averaged over its planning value's Monte Carlo error.

    Parameters
    ----------
    row : str
        A key of :data:`PLANNING`.
    replicates : int
        The budget.

    Returns
    -------
    float
        The predictive pass probability.
    """
    point, spread, _ = PLANNING[row]
    draws = point + spread * _GH_X
    judge = beyond_pass if row == "fold_locality/ate__pooled" else bias_pass
    return float(np.sum(_GH_W * judge(draws, replicates)))


def displacement_pass(replicates: int) -> float:
    """The probability that the paired displacement reaches its threshold, under P1's value."""
    mean, spread = LEAK_PAIRED
    planning_se = 0.025
    standard = np.hypot(spread / math.sqrt(replicates), planning_se)
    return float(norm.cdf((mean - FOLD_LOCALITY_DISPLACEMENT) / standard))


#: The units of the study-wise rule: each fit group, its gated rows, and its budget floor.
UNITS: Mapping[str, tuple[tuple[str, ...], int]] = {
    "primary": (tuple(row for row in PLANNING if row.startswith("primary/")), PRIMARY_FLOOR),
    "fold_locality": (("fold_locality/ate__foldwise", "fold_locality/ate__pooled"), PROPERTY_FLOOR),
    "benkeser_order": (("benkeser_order/ate",), PROPERTY_FLOOR),
    "bivariate": (("bivariate/ate",), PROPERTY_FLOOR),
    "k5": (("k5/ate",), PROPERTY_FLOOR),
    "adaptive_reduced": (("adaptive_reduced/ate",), PROPERTY_FLOOR),
    "multi_arm": (("multi_arm/ate",), PROPERTY_FLOOR),
    "stratified": (
        tuple(row for row in PLANNING if row.startswith("stratified/")),
        PROPERTY_FLOOR,
    ),
}
UNIT_TARGET = STUDY_TARGET ** (1.0 / len(UNITS))


def _point(row: str, replicates: int) -> float:
    point, _, _ = PLANNING[row]
    judge = beyond_pass if row == "fold_locality/ate__pooled" else bias_pass
    return float(judge(point, replicates))


def point_pass(unit: str, replicates: int) -> float:
    """A unit's pass probability at the planning points, rows taken independent.

    Rows of one fit are positively correlated, so the product is a lower bound.  The leak
    unit also needs its paired displacement.
    """
    rows, _ = UNITS[unit]
    value = float(np.prod([_point(row, replicates) for row in rows]))
    if unit == "fold_locality":
        value *= displacement_pass(replicates)
    return value


def unit_pass(unit: str, replicates: int) -> float:
    """A unit's predictive pass probability, averaged over the planning values' error.

    Reported beside the budgets; the budgets follow :func:`point_pass`.
    """
    rows, _ = UNITS[unit]
    value = float(np.prod([predictive(row, replicates) for row in rows]))
    if unit == "fold_locality":
        value *= displacement_pass(replicates)
    return value


@cache
def budget(unit: str) -> tuple[int, float, float]:
    """``(R, point pass, predictive pass)``: the smallest grid budget whose point pass reaches
    the unit target.
    """
    _, floor = UNITS[unit]
    for replicates in range(floor, CEILING + 1, STEP):
        value = point_pass(unit, replicates)
        if value >= UNIT_TARGET:
            return replicates, value, unit_pass(unit, replicates)
    raise ValueError(f"{unit} does not reach the unit target by R = {CEILING}")


def study_pass(*, point: bool = False) -> float:
    """The study-wise pass probability at the declared budgets, predictive unless ``point``."""
    judge = point_pass if point else unit_pass
    return float(np.prod([judge(unit, BUDGETS[unit]) for unit in UNITS]))


#: The declared budgets, each the output of :func:`budget`.
BUDGETS: Mapping[str, int] = {
    "primary": 2_000,
    "fold_locality": 4_400,
    "benkeser_order": 2_000,
    "bivariate": 2_000,
    "k5": 2_000,
    "adaptive_reduced": 2_000,
    "multi_arm": 3_200,
    "stratified": 1_600,
}

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
        The group's name; it labels the sample stream.
    law : str
        ``"paper"`` or ``"stratified"``.
    n : int
        The sample size.
    fits : tuple of tuple
        ``(family, cell, role, Fit)`` for each fit made on the sample.
    """

    name: str
    law: str
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


GROUPS: Mapping[str, Group] = {
    "fold_locality": Group(
        "fold_locality",
        "paper",
        LEAK_N,
        (
            ("fold_locality", "ate__foldwise", "positive", Fit()),
            ("fold_locality", "ate__pooled", "control", Fit(reduced_crossfit=study.POOLED)),
        ),
    ),
    "benkeser_order": Group(
        "benkeser_order",
        "paper",
        PAPER_N,
        (("double_robustness", "ate__benkeser_order", "positive", Fit(update_order="benkeser")),),
    ),
    "bivariate": Group(
        "bivariate",
        "paper",
        PAPER_N,
        (("double_robustness", "ate__bivariate", "positive", Fit(reduction="bivariate")),),
    ),
    "k5": Group(
        "k5", "paper", PAPER_N, (("double_robustness", "ate__k5", "positive", Fit(n_folds=5)),)
    ),
    "adaptive_reduced": Group(
        "adaptive_reduced",
        "paper",
        PAPER_N,
        (("double_robustness", "ate__adaptive_reduced", "positive", Fit(reduced="spline")),),
    ),
    "stratified": Group(
        "stratified",
        "stratified",
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

    Every setting is that study's ``fit_cleverly`` except ``reduced_crossfit``.
    """
    outcome, treatment = stratified.learners(STRATIFIED_SCENARIO)
    return (
        stratified.FixedFoldDRTMLE(
            frame["fold"].to_numpy(dtype=np.int64),
            outcome_learner=outcome,
            treatment_learner=treatment,
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=stratified.ColumnLogistic(),
            cross_fit=True,
            n_folds=stratified.N_FOLDS,
            stratify_folds="none",
            estimands=("ey", "ate"),
            simultaneous=False,
            g_bounds=stratified.G_BOUNDS,
            max_outer=stratified.MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            guard=("Q", "g"),
            reduced_crossfit=fit.reduced_crossfit,
        )
        .fit(
            frame, outcome="Y", treatment="A", covariates=list(stratified.COVARIATES), strata=["V"]
        )
        .single()
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
    return study.draw_paper(SCENARIO, group.n, seed, group.n_folds)


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
        result = study.fit_cleverly(frame, SCENARIO, fit)
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
    """Fit every property replication; ``budget`` caps each loop for a smoke run."""
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
