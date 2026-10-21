r"""Registered evidence for the structural nested mean estimator (``structural-nested-mean``, X44).

The subject is :class:`~cleverly.longitudinal.StructuralNestedMean` with declared
:class:`~cleverly.learners.GLM` nuisances (and an early-stopped booster on the nonlinear law
``N``).  The laws are in :mod:`tests.studies.structural_nested_mean_laws` and the property
families in :mod:`tests.studies.structural_nested_mean_properties`.

The comparator is EconML 0.17.0 ``DynamicDML``, registered as :data:`STUDY`'s ``reference``
(:data:`COMPARATOR_STUDY`, the record the registry reads).  :data:`PENDING_STUDY` is the same
record before registration, which the registry tests use.  The comparator fits two variants in
one container run (``tests/canonical/econml_dynamic_dml/run_study.py``).

* The *paired* variant fits :data:`COMPARATOR_PAIRED` on the subject's rows, folds and ``GLM``
  design, with the full history as the period state.  Its rows pair with the subject's on the
  cells of :data:`REFERENCE_SCENARIOS`, the cells ``DynamicDML`` reports.
* The *Markov* variant fits :data:`COMPARATOR_MARKOV` as the published estimator states it, with
  the period state ``(W, S_t, A_{t-1})``.  Its rows go to a descriptive table beside the computed
  limit of each cell (:func:`markov_summary`).  The table gates nothing.

The registered run declares :data:`COMPARATOR_REPLICATES`, sized from
:data:`COMPARATOR_PILOT`.  :data:`PRIMARY_REPLICATES` equals it.

**Primary scenarios.**  One scenario per positive setup of
:func:`tests.studies.structural_nested_mean_laws.setups` at :data:`PRIMARY_N` with
:data:`PRIMARY_REPLICATES` replicates.

========  ======================================================================================
scenario  what it shows
========  ======================================================================================
``s2``    two binary nodes, ``phi = 1``
``s3``    three binary nodes with an :math:`A_t \to A_j` path
``k``     two nodes of three levels
``c``     two doses; every conditional mean is linear in the history
``h``     :math:`\phi_1 = (1, W)`, :math:`\phi_2 = (1, W, A_1)`
``p``     a past-treatment feature, :math:`\phi_2 = (1, A_1)`
``m``     the second treatment and the outcome read the first covariate
``l``     :math:`\phi_2 = (1, S_2)`: the blips and ``ey_reference`` (no static contrast)
``n``     nonlinear nuisances, a cross-fitted early-stopped booster
``s2w``   known weights
``s2c``   clusters of ten
========  ======================================================================================

**Negative controls.**  ``LU`` is law ``L`` fitted with :math:`\phi_2 = 1`, and ``VB`` is law
``P`` fitted with a baseline-only :math:`\phi_2`.  Their regimen contrasts miss the regimen
truth.  They are property cells (family ``blip_feature_control``), not primary scenarios: the
replication check refuses a primary cell with coverage below 50% as a broken join.  A control
reads as a detection rule, as every other control of the instrument does.  A cell passes when
its 99% standardized-bias interval lies beyond 0.25 SD on the side of the computed gap, so a
green row shows that the instrument sees the misspecified blip.  A control that stays inside the
margin is a red row: the instrument is blind to the misspecification.  The roadmap's "truth row
owned by the structural label" therefore reads as a green detection row.

**The model check.**  ``check_model`` reads the joint Wald statistic against a Hotelling law
whose degrees of freedom are the Welch-Satterthwaite value of the difference covariance.  On the
sparse-positivity law ``SP`` the median covariance has about 8 degrees of freedom, so the check
rejects a misspecified model with low power there, and the size cells at n = 2,000 measure the
size only.  The page says so beside the cells.  On ``S2`` and ``H`` the check rejects less often
than the nominal rate, so it is conservative.  The rejection rate is published beside every
verdict.

**Comparator reading (declared before the run).**  The chances come from the 1,500-replicate
pilot of ``k`` (:data:`COMPARATOR_PILOT`) at :data:`COMPARATOR_REPLICATES` replicates.  Each
item gives the row, the expected verdict, the chance, and the route if the row reads otherwise.

* *21 paired rows.*  Expected: pass, with chance 0.958 together.  The items below give the
  routes.
* *``k`` ``ate_regimen[r12 vs r00]``.*  Expected: ``superior``, with chance 0.87, and otherwise
  ``equivalent``.  Both read as a pass, so no route is needed.
* *``k`` ``ate_regimen[r21 vs r00]``.*  Expected: ``equivalent``, with chance 0.958.
  ``inconclusive`` on the coverage leg is the reference's larger SE: the mean ratio is 1.031,
  from the transposed categorical blocks (``test_w2_*`` in
  ``tests/unit/test_structural_econml_parity.py``).  The subject's own calibration cell reads
  1.0016 at R = 8,000.  Publish the row red under ``reporting`` with a named owner row.  No
  re-run and no added replicates.
* *The other 19 paired rows.*  Expected: ``equivalent``, with chance about 1.  A red row is a
  defect suspect: diagnose it first.
* *The 21 ``econml`` truth rows.*  Expected: pass, with the ``k`` ``r12`` SE ratio about 0.975
  and ``r21`` about 1.026, and chance about 1.  A red ``econml`` row is a reference finding.  The
  subject row stays as it is.
* *The 61 subject truth rows.*  Expected: pass, and replicates 0 to 999 equal the committed rows
  bit for bit, with chance about 1.  Any difference in replicates 0 to 999 stops the reading:
  the code moved.
* *``exactness.csv``.*  Expected: every gap under :data:`EXACTNESS_GATE`, with chance about 1.  A
  red row is a defect suspect.
* *The Markov table, 26 cells.*  Expected: ``agrees_with_limit`` true, with truth coverage near 0
  on the gap cells (``s3``, ``m``, ``p``, ``h``) and near 0.95 on ``s2`` and ``k``.  The table
  is descriptive and gates nothing.  A cell off its limit is a finding about the limit
  computation, not about either estimator.

``superior`` on ``k`` ``r12`` comes from the reference's SE defect.  It does not show that the
subject estimates better, because the points are identical.

**Red-cell reading** (declared before any run).  The study publishes under ``reporting``.  No
row is declared red: :data:`EXPECTED_RED` is empty.  The sparse-positivity law ``SP`` was
declared red in an earlier draft on the reading that the unpenalized ``GLM`` nuisances fail in
finite samples.  A probe showed no fit failure at n = 2,000, so the draft was wrong, and
``check_model`` now uses the Satterthwaite reference.  The size cells of ``SP`` are positive
cells at n = 2,000.  The limited power of the check on ``SP`` is a statement about the cell, not
a red row.  The under-coverage of ``LTMLE`` for a low-probability regimen under weak positivity
is a property of ``LTMLE``, and this study measures ``StructuralNestedMean``.

A red row is a defect suspect.  It is a finite-sample red only when
the 99% interval of its SE ratio contains 1, its coverage is consistent with 0.95 and the
diagnosis finds no defect; it then goes to ``reporting`` with an owner, declared before any repeat
run.  A defect is fixed and the study re-runs under a fresh declaration.  A replicate that raises a
recordable error is recorded and never redrawn; the declared failure bound is
:data:`FAILURE_BOUND` per cell.  No budget, margin, law, learner or seed moves after a verdict is
seen.
"""

from __future__ import annotations

import dataclasses
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleverly.learners import GLM
from cleverly.learners.crossfit import random_partition
from cleverly.longitudinal import StructuralNestedMean
from tests.parallel import STUDY_JOBS
from tests.studies import structural_nested_mean_laws as laws
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SEED = 20261444
RESAMPLING_SEED = 2026144401

#: The size of every primary scenario and of every calibration cell.
PRIMARY_N = 2_000
CALIBRATION_N = PRIMARY_N
#: The replicates of every primary scenario: :data:`COMPARATOR_REPLICATES`, the count the
#: predictive rule gives for the comparator's paired legs (a unit test pins the two together).
PRIMARY_REPLICATES = 1_600
#: The three sizes of the ladder.  Every rung is a positive cell (pilot: ``PILOT_LADDER`` in
#: ``tests/unit/test_structural_nested_mean_design.py``).
LADDER = (500, 2_000, 8_000)
#: The ladder of a setup that does not use :data:`LADDER`.  ``K`` loses 1.3% of its draws to a
#: separated cell at n = 500 (the three-level product indicators of the second treatment), above
#: the failure bound, so its ladder keeps the factor-4 spacing from n = 1,000.
LADDER_SIZES: dict[str, tuple[int, ...]] = {"K": (1_000, 4_000, 16_000)}


def ladder_sizes(setup: str) -> tuple[int, ...]:
    """The sizes of a setup's ladder."""
    return LADDER_SIZES.get(setup, LADDER)


#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01

IMPLEMENTATION = "cleverly-structural-nested-mean"

#: The comparator: EconML 0.17.0 ``DynamicDML`` (``tests/canonical/econml_dynamic_dml``).
REFERENCE = "econml"

# ------------------------------------------------------------------------------- the fits

#: The ``GLM`` interaction order of each setup's nuisances: the lowest order that returns every
#: truth at population weights (the design test checks it, and that one order less fails except
#: at main terms).  ``3`` unless listed.  ``N`` has no entry: it fits a booster.  An order of
#: three on ``S2`` fails on a separated cell in 9.7% of the draws at n = 500.  The order applies to
#: the outcome regressions, and to the treatment regressions unless
#: :data:`TREATMENT_INTERACTIONS` lists the setup.
INTERACTIONS: dict[str, int] = {
    "S2": 2,
    "H": 2,
    "P": 2,
    "L": 2,
    "K": 2,
    "S2W": 2,
    "S2C": 2,
    "C": 1,
    "LU": 2,
    "VB": 2,
}
#: The treatment-side ``GLM`` order of a setup whose two roles need different orders.  The node-2
#: outcome regression of ``SP`` needs a three-way cell, so its outcome order is three.  An order
#: of three on both roles failed on a separated treatment cell in 17% of the draws at n = 2,000
#: (101 of 600).  The treatment logits of ``SP`` are additive, so its treatment regressions are
#: exact at any order; order two is the order of the model-check fits
#: (:data:`CHECK_INTERACTIONS`).  The declared pair fails in none of 1,000 draws
#: (``PILOT_POSITIVITY`` in the design test).
TREATMENT_INTERACTIONS: dict[str, int] = {"SP": 2}
DEFAULT_INTERACTIONS = 3
#: The outer folds of each setup; ``10`` (the shipped default) unless listed.
DEFAULT_FOLDS = 10
FOLDS: dict[str, int] = {"N": 5}


#: The settings of ``N``'s boosters.  Their capacity grows with n: each fit holds out
#: ``validation_fraction`` of its training rows and stops when the held-out loss has not improved
#: for ``n_iter_no_change`` rounds, so the number of trees is chosen by the data.
#:
#: Why this learner.  The structural estimator's remainder is a product of nuisance errors, so
#: nuisances that converge faster than :math:`n^{-1/4}` make it negligible (Chernozhukov et al.
#: 2018, Thm 3.1).  A booster of fixed size has a fixed bias, and the round-2 review measured it:
#: ``blip[A2]`` read 1.8 SE of bias at n = 512,000.  Early stopping on a held-out split is
#: hold-out model selection over a nested sieve (the number of rounds).  The selected fit attains
#: the rate of the best stopping time up to an :math:`O(\log n / n)` term (Gyorfi, Kohler,
#: Krzyzak and Walk 2002, Thm 7.1), and boosting with early stopping is consistent (Zhang and Yu
#: 2005).  Trees of 15 leaves with no depth cap reach every interaction of the five history
#: variables, so the sum of trees can represent each nuisance of the finite law.  A depth of two
#: cannot represent :math:`q_2 = E[Y \mid H_2]`, which holds the ``expit`` of the node-2 logit, an
#: interaction of three variables.  A cross-validated library was the alternative.  Its oracle
#: inequality only carries over the rate of its best candidate, so it would still need a
#: candidate of growing capacity, at the cost of an inner cross-validation.  ``max_iter`` is a cap
#: that the stopping rule never reaches at the probed sizes (``BOOSTER_ROUNDS`` in the design
#: test).
BOOSTER_SETTINGS: dict[str, Any] = {
    "max_depth": None,
    "max_leaf_nodes": 15,
    "learning_rate": 0.1,
    "max_iter": 2_000,
    "l2_regularization": 1.0,
    "min_samples_leaf": 20,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 0,
}


def booster_classifier() -> HistGradientBoostingClassifier:
    """The early-stopped booster of ``N``'s treatment mechanisms."""
    return HistGradientBoostingClassifier(**BOOSTER_SETTINGS)


def booster_regressor() -> HistGradientBoostingRegressor:
    """The early-stopped booster of ``N``'s outcome regressions."""
    return HistGradientBoostingRegressor(**BOOSTER_SETTINGS)


#: The ``GLM`` interaction order of the model-check fits, for both roles.  At n = 1,000 and two
#: folds an order of three fails on a separated follower cell in about 14% of the draws of
#: ``S2``; an order of two fails in none of 150.  On ``SP`` it leaves the node-2 outcome
#: regression inexact; the treatment regressions stay exact, so the difference of the two
#: estimators still has mean zero under a correct model by double robustness.
CHECK_INTERACTIONS = 2
#: The splits of the model check (``check_model(n_splits=...)``): the
#: shipped default.
CHECK_SPLITS = 5


def learners(name: str, order: int | None = None) -> dict[str, Any]:
    """The outcome and treatment learners of one setup; ``order`` replaces both ``GLM`` orders."""
    if laws.setup(name).nuisance == "booster":
        return {
            "outcome_learner": booster_regressor(),
            "treatment_learner": booster_classifier(),
        }
    if order is None:
        outcome = INTERACTIONS.get(name, DEFAULT_INTERACTIONS)
        treatment = TREATMENT_INTERACTIONS.get(name, outcome)
    else:
        outcome = treatment = order
    return {
        "outcome_learner": GLM(family="gaussian", interactions=outcome),
        "treatment_learner": GLM(interactions=treatment),
    }


#: The learner class of each nuisance role of a ``GLM`` setup and of ``N``.
GLM_LEARNERS: dict[str, type] = {"outcome": GLM, "treatment": GLM}
BOOSTER_LEARNERS: dict[str, type] = {
    "outcome": HistGradientBoostingRegressor,
    "treatment": HistGradientBoostingClassifier,
}


@dataclass(frozen=True)
class SupportDeclaration:
    """The RM42b support key one setup's fits record, with the learner class of each role.

    Every key of this study also reads the ``snmm`` engine, ``cross_fitted`` splitting, a
    gaussian unbounded outcome, complete data and the ``rademacher`` multiplier.

    Parameters
    ----------
    treatment : tuple of str
        The level count of each treatment node, or ``"continuous"``.
    weighting : str
        ``"none"`` or ``"fixed"`` (known weights).
    dependence : str
        ``"iid"`` or ``"clustered"``.
    covariate_features : bool
        Whether a blip feature reads a time-varying covariate.
    regimens : bool
        Whether the fit declares static regimens.
    learners : dict
        The learner class of the ``outcome`` and ``treatment`` roles.
    """

    treatment: tuple[str, ...] = ("2", "2")
    weighting: str = "none"
    dependence: str = "iid"
    covariate_features: bool = False
    regimens: bool = True
    learners: Mapping[str, type] = field(default_factory=lambda: dict(GLM_LEARNERS))


#: The support key each fitted setup records under ``CLEVERLY_SUPPORT_RECORD``.  The design test
#: fits every setup under the record switch and requires the recorded key and learner classes
#: to equal this declaration, so the support rows written after the run name what the run
#: measured.  The ``N`` key names the booster classes; its claim is the early-stopped booster of
#: :func:`booster_regressor` and :func:`booster_classifier`.  The model check also records the
#: key of its ``LTMLE`` side.  A model-check row is supported only when that ``LTMLE`` key has
#: its own row.
SUPPORT_KEYS: dict[str, SupportDeclaration] = {
    "S2": SupportDeclaration(),
    "S3": SupportDeclaration(treatment=("2", "2", "2")),
    "K": SupportDeclaration(treatment=("3", "3")),
    "C": SupportDeclaration(treatment=("continuous", "continuous")),
    "H": SupportDeclaration(),
    "P": SupportDeclaration(),
    "M": SupportDeclaration(),
    "L": SupportDeclaration(covariate_features=True, regimens=False),
    "N": SupportDeclaration(learners=dict(BOOSTER_LEARNERS)),
    "S2W": SupportDeclaration(weighting="fixed"),
    "S2C": SupportDeclaration(dependence="clustered"),
    "SP": SupportDeclaration(),
    "LU": SupportDeclaration(),
    "VB": SupportDeclaration(),
}


class StatusMismatch(AssertionError):
    """A fit stamped an inference status that its cell does not read.

    An ``AssertionError`` is not a recordable fitting failure
    (:func:`cleverly.validation._failures.recordable`), so it stops the run.  A replicate row
    reads the reported standard error under any status, so a silently degraded status would
    calibrate the wrong curve, and the rows could not show it afterwards.
    """


#: The status of every fit this study reads.
INFERENTIAL = "influence_curve"


def require_status(result: Any, names: Any, expected: str = INFERENTIAL) -> None:
    """Raise :class:`StatusMismatch` unless the fit and each read parameter carry ``expected``."""
    if result.inference_status != expected:
        raise StatusMismatch(f"fit status {result.inference_status!r}, expected {expected!r}")
    for name in names:
        estimate = result[name]
        if estimate.inference != expected or not bool(estimate.supplies_inference):
            raise StatusMismatch(
                f"{name}: status {estimate.inference!r} (supplies inference "
                f"{estimate.supplies_inference}), expected {expected!r}"
            )


def fit(
    name: str,
    frame: pd.DataFrame,
    *,
    folds: int | None = None,
    clustered: bool | None = None,
    order: int | None = None,
    random_state: int = 0,
) -> Any:
    """One fit of the estimator on a draw of setup ``name``.

    ``folds`` replaces the setup's outer folds.  ``clustered=False`` fits a clustered draw
    without its cluster column (the unclustered diagnostic).  A setup with no declared regimen
    (``L``, whose feature reads a covariate) reports the blips and ``ey_reference`` only.  ``order``
    replaces the ``GLM`` interaction order.
    """
    chosen = laws.setup(name)
    law = chosen.law
    use_cluster = chosen.cluster if clustered is None else clustered
    estimator = StructuralNestedMean(
        chosen.regimens or None,
        reference=chosen.reference or None,
        blip_features=chosen.features,
        n_folds=(FOLDS.get(name, DEFAULT_FOLDS) if folds is None else folds),
        simultaneous=False,
        random_state=random_state,
        **learners(name, order),
    )
    with warnings.catch_warnings():
        # The shipped warnings (positivity, extrapolation) are diagnostics; none changes a
        # number, and the design test reads the positivity share directly.
        warnings.simplefilter("ignore")
        return estimator.fit(
            frame,
            outcome="Y",
            treatment=law.treatments,
            baseline=law.baseline,
            time_varying=law.time_varying,
            id="id" if use_cluster else None,
            weights="w" if chosen.weighted else None,
            family="gaussian",
            continuous_treatment=list(law.doses),
        )


# --------------------------------------------------------------------------- scenarios

#: The negative controls: their regimen contrasts are biased by construction.  They are
#: property cells, not primary scenarios, because the replication check refuses a primary cell
#: whose coverage is below 50% as a broken join.
NEGATIVE_CONTROLS = ("LU", "VB")
#: Scenario name to the setup it fits.
SETUP_OF: dict[str, str] = {
    name.lower(): name for name in laws.setups() if name not in {"SP", *NEGATIVE_CONTROLS}
}
SCENARIOS: dict[str, tuple[str, ...]] = {
    scenario: laws.setup(name).estimands for scenario, name in SETUP_OF.items()
}
SEED_OWNERS: dict[str, str] = {scenario: scenario for scenario in SCENARIOS}
#: The replicates of every negative-control cell, sized by the design test (each cell must fail
#: the calibration rule with chance at least 0.95).
CONTROL_REPLICATES = 2_000

# ---------------------------------------------------------------------------- the labels

#: The positive setups whose estimands are calibrated.
CALIBRATED = ("S2", "S3", "K", "C", "H", "P", "M", "L", "N", "S2W", "S2C")
#: The ladder estimands of each laddered setup.  ``N`` is the adaptive cell: a contrast and both
#: blips, so a remainder that does not vanish shows in three estimands.  ``blip[A2]`` carried
#: the largest bias of the fixed booster that the round-2 review measured.
LADDER_ESTIMANDS: dict[str, tuple[str, ...]] = {
    "S2": ("ate_regimen[r11 vs r00]",),
    "H": ("ate_regimen[r11 vs r00]",),
    "K": ("ate_regimen[r12 vs r00]",),
    "C": ("ate_regimen[dose_high vs dose0]",),
    "N": ("ate_regimen[r11 vs r00]", "blip[A1:1.0:intercept]", "blip[A2:1.0:intercept]"),
}


def short(estimand: str) -> str:
    """The short key of an estimand: ``blip_a1_l1_w``, ``ey0``, ``ey_r11`` or ``ate_r11``."""
    if estimand == "ey_reference":
        return "ey0"
    match = re.fullmatch(r"blip\[(\w+):([\w.]+):(\w+)\]", estimand)
    if match:
        treatment, level, feature = match.groups()
        parts = ["blip", treatment.lower()]
        if level != "dose":
            parts.append(f"l{int(float(level))}")
        if feature != "intercept":
            parts.append(feature.lower())
        return "_".join(parts)
    match = re.fullmatch(r"ey_regimen\[(\w+)\]", estimand)
    if match:
        return f"ey_{match.group(1)}"
    match = re.fullmatch(r"ate_regimen\[(\w+) vs \w+\]", estimand)
    if match:
        return f"ate_{match.group(1)}"
    raise KeyError(estimand)


def label_of(setup: str, estimand: str) -> str:
    """The calibration label of one estimand of one setup."""
    return f"{setup.lower()}_{short(estimand)}"


#: ``label -> (setup, estimand)`` for every calibration label.
TARGETS: dict[str, tuple[str, str]] = {
    label_of(setup, estimand): (setup, estimand)
    for setup in CALIBRATED
    for estimand in laws.setup(setup).estimands
}
CALIBRATION_LABELS: tuple[str, ...] = tuple(TARGETS)
#: The negative-control targets, one per estimand of ``LU`` and ``VB``.
CONTROL_TARGETS: dict[str, tuple[str, str]] = {
    label_of(setup, estimand): (setup, estimand)
    for setup in NEGATIVE_CONTROLS
    for estimand in laws.setup(setup).estimands
}
CONTROL_LABELS: tuple[str, ...] = tuple(CONTROL_TARGETS)
#: ``label -> (setup, estimand)`` for every ladder label.
LADDER_TARGETS: dict[str, tuple[str, str]] = {
    label_of(setup, estimand): (setup, estimand)
    for setup, estimands in LADDER_ESTIMANDS.items()
    for estimand in estimands
}
LADDER_LABELS: tuple[str, ...] = tuple(LADDER_TARGETS)
#: The weak-positivity reporting cell: every estimand of ``SP`` at n = 2,000.  It states the
#: regime in which the structural estimator is fitted under treatment probabilities of 0.05 to
#: 0.94, and gates nothing.
POSITIVITY_N = 2_000
POSITIVITY_REPLICATES = 1_000
POSITIVITY_TARGETS: dict[str, tuple[str, str]] = {
    label_of("SP", estimand): ("SP", estimand) for estimand in laws.setup("SP").estimands
}
POSITIVITY_LABELS: tuple[str, ...] = tuple(POSITIVITY_TARGETS)


def truth_of(label: str) -> float:
    """The exact truth of one calibration label."""
    setup, estimand = TARGETS[label]
    return laws.truth(setup, estimand)


#: The sizes of the model-check cells and their declared replicates, sized by the design test.
SIZE_N = 1_000
#: The size cells at the power cell's n, so a high ``LU`` rejection rate reads against a size
#: measured at the same n and folds.
POWER_N = 2_000
#: The sparse-positivity size cells.  At n = 500 the declared ``GLM`` fails the failure rule: the
#: pilot lost 19% of the draws to a separated cell.  At n = 2,000 no fit fails.
SPARSE_N = 2_000
SIZE_REPLICATES = 600
POWER_REPLICATES = 200
#: The check cells: ``name -> (kind, setup, n, folds)``.
CHECK_CELLS: dict[str, tuple[str, str, int, int]] = {
    "s2_f2": ("size", "S2", SIZE_N, 2),
    "s2_f10": ("size", "S2", SIZE_N, 10),
    "h_f2": ("size", "H", SIZE_N, 2),
    "h_f10": ("size", "H", SIZE_N, 10),
    "s2_n2000_f2": ("size", "S2", POWER_N, 2),
    "s2_n2000_f10": ("size", "S2", POWER_N, 10),
    "h_n2000_f2": ("size", "H", POWER_N, 2),
    "h_n2000_f10": ("size", "H", POWER_N, 10),
    "sp_f2": ("size", "SP", SPARSE_N, 2),
    "sp_f10": ("size", "SP", SPARSE_N, 10),
    "lu_f2": ("power", "LU", POWER_N, 2),
    "lu_f10": ("power", "LU", POWER_N, 10),
}

#: The replicates of a check cell that differ from :data:`SIZE_REPLICATES` and
#: :data:`POWER_REPLICATES`.  The sparse-positivity cell at ten folds rejects 6.0% in the pilot (2.8% at
#: two folds), so the size rule needs 1,200 replicates there for a chance of 0.90 or more
#: (``tests/unit/test_structural_nested_mean_design.py``).
CHECK_REPLICATES: dict[str, int] = {"sp_f10": 1_200}


def check_replicates(name: str) -> int:
    """The declared replicates of one model-check cell."""
    kind = CHECK_CELLS[name][0]
    default = SIZE_REPLICATES if kind == "size" else POWER_REPLICATES
    return CHECK_REPLICATES.get(name, default)


#: Every published property cell, by family.  :mod:`tests.studies.structural_nested_mean_properties`
#: declares how each is drawn and read.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "root_n_and_efficiency": tuple(
        f"{label}__n_{size}"
        for label, (setup, _) in LADDER_TARGETS.items()
        for size in ladder_sizes(setup)
    ),
    "root_n_rate": tuple(
        f"{label}__{suffix}"
        for label in LADDER_LABELS
        for suffix in ("empirical_sd", "reported_se")
    ),
    "warning_rate": tuple(
        f"{name}__model_check" for name, (kind, *_) in CHECK_CELLS.items() if kind == "size"
    ),
    "power": tuple(
        f"{name}__model_check" for name, (kind, *_) in CHECK_CELLS.items() if kind == "power"
    ),
    "positivity_reporting": tuple(f"{label}__weak_positivity" for label in POSITIVITY_LABELS),
    "unclustered_se": tuple(
        f"{label_of('S2C', estimand)}__unclustered" for estimand in laws.setup("S2C").estimands
    ),
    "blip_feature_control": tuple(f"{label}__too_few_features" for label in CONTROL_LABELS),
}

#: The property rows declared red: none.  The negative controls read as a detection rule and
#: pass when the instrument sees the misspecification, and the sparse-cell size cells are
#: positive cells at n = 2,000.
EXPECTED_RED_PROPERTIES: tuple[str, ...] = ()
EXPECTED_RED: tuple[str, ...] = EXPECTED_RED_PROPERTIES

# ------------------------------------------------------------------------- the comparator

#: What the manifest records about the comparator: the version, the pinned base image and the
#: hash of the lock file.  ``tests/unit/test_structural_nested_mean_comparator.py`` reads the same
#: values back from the Dockerfile and ``requirements.txt``, so the three cannot drift.
REFERENCE_METADATA = {
    "econml_version": "0.17.0",
    "python_base_image": (
        "python:3.13-slim-bookworm@sha256:"
        "a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641"
    ),
    "requirements_sha256": "db82abfde18ad9373bc540c15863b0905b6b3a81e0946988bd0fb79c985cd10f",
}
#: The paired scenarios.  ``DynamicDML`` fits each one on the subject's rows and folds, with the
#: full history as the period state and the subject's ``GLM`` design, so its paired rows measure
#: one estimand by one estimating equation.  The pilot of ``k`` (:data:`COMPARATOR_PILOT`) read
#: points equal within 2.4e-14 at most, and a 3-replicate run of every paired scenario read them
#: within 3.8e-14.  For a scalar node the standard errors are in the ratio
#: :math:`\sqrt{(n - 1) / n}`.  ``exactness.csv`` (:func:`extra_artifacts`) gates the largest gap
#: of each cell against :data:`EXACTNESS_GATE`.
#:
#: ``n`` is left out: a booster chooses its validation split from the column layout of the long
#: panel, which differs from the subject's frame, so its points would not be exact and a pairing
#: on the margins alone would compare two different fits.  ``l`` is left out: ``DynamicDML`` has
#: no blip feature that reads a covariate.  ``s2w`` is left out: ``DynamicDML.fit`` accepts
#: ``sample_weight`` and ignores it with a ``UserWarning``, so it cannot state the weights.
#: ``s2c`` is left out: ``DynamicDML`` has no cluster variance.  ``h`` and ``p`` are left out: a
#: blip of theirs reads ``A_1``, which the features ``X`` of ``DynamicDML`` cannot state.  Their
#: Markov rows are published instead.
COMPARATOR_PAIRED = ("s2", "s3", "k", "c", "m")
#: The native scenarios: the Markov-state fit of the published estimator.  Their rows go to a
#: descriptive table beside the computed limits
#: (:func:`tests.studies.structural_nested_mean_laws.dynamic_dml_limit`), and gate nothing.
COMPARATOR_MARKOV = ("s2", "s3", "k", "h", "p", "m")
COMPARATOR_SCENARIOS = tuple(sorted({*COMPARATOR_PAIRED, *COMPARATOR_MARKOV}))
#: The blip features ``X`` of a Markov fit; ``phi = 1`` unless listed.
MARKOV_FEATURES: dict[str, tuple[str, ...]] = {"h": ("W",)}
#: The ``GLM`` order that saturates a Markov state ``(W, S_t, A_{t-1})`` of binary columns, and
#: of ``K``, whose previous treatment has two indicator columns.
MARKOV_ORDERS: dict[str, int] = {"binary": 3, "categorical": 4}
#: The pilot of the paired coverage leg on ``k``: 1,500 replicates (indices 5,000 to 6,499, outside
#: every declared run) at n = 2,000, both sides on the same rows and folds.  Each entry is
#: ``(replicates, discordant pairs, the side that covers in each of them)``.  The points agree
#: within 2.4e-14 in every cell, and the blips' standard errors agree within 0.3%, so no blip pair
#: is discordant.  The EconML contrast errors differ from the subject's by the mean ratios 0.975
#: (``r12``) and 1.031 (``r21``); the categorical blocks of its sandwich are transposed
#: (``tests/canonical/econml_dynamic_dml/README.md``).
COMPARATOR_PILOT: dict[str, tuple[int, int, str]] = {
    "ate_regimen[r12 vs r00]": (1_500, 8, "subject"),
    "ate_regimen[r21 vs r00]": (1_500, 15, "reference"),
}
#: The replicates the registered comparator run declares, from the predictive rule of
#: ``tests/unit/test_structural_nested_mean_comparator.py``: the smallest multiple of 200 at
#: or above :data:`PRIMARY_REPLICATES` at which every paired cell passes with chance at least
#: 0.90 and the family with chance at least 0.95.  The binding cell is the coverage leg of
#: ``k``'s ``r21`` contrast, where the reference covers more often because its error is larger.
#: :data:`PRIMARY_REPLICATES` is this count, so every primary scenario runs at it and the paired
#: rows and the subject's own rows share replicates 0 to 1,599.
COMPARATOR_REPLICATES = 1_600
#: The exactness gate: a paired cell passes when its largest estimate gap is below this.  The
#: pilot reads gaps near 1e-14, so a gap of 1e-9 is a different estimating equation.
EXACTNESS_GATE = 1e-9
#: The seed of the subject's folds: :func:`fit` passes ``random_state=0``.  The paired sample
#: carries the fold of each row, and the fit checks it against ``result.folds``.
COMPARATOR_FOLD_SEED = 0
#: The task-table columns of ``tests/canonical/econml_dynamic_dml/run_study.py``.
TASK_COLUMNS = (
    "scenario",
    "variant",
    "estimand",
    "truth",
    "kind",
    "nodes",
    "order",
    "features",
    "t1",
    "t0",
)
#: The descriptive Markov artefacts.
MARKOV_REPLICATES = "econml-markov-replicates.csv.gz"
MARKOV_SUMMARY = "econml-markov.csv"
MARKOV_FAILURES = "econml-markov-failures.csv"
#: The gated table of the paired cells' largest estimate gaps.
EXACTNESS_FILE = "exactness.csv"
#: A Markov cell agrees with its computed limit when the 99% interval of its standardized
#: distance from the limit lies inside this margin (the similarity margin of :class:`Margins`).
MARKOV_MARGIN = 0.25


def treatment_kind(setup: str) -> str:
    """``"binary"``, ``"categorical"`` or ``"continuous"``: the treatment type of every node."""
    levels = set(SUPPORT_KEYS[setup].treatment)
    if levels == {"continuous"}:
        return "continuous"
    if levels == {"2"}:
        return "binary"
    return "categorical"


def comparator_estimands(setup: str, features: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The estimands of a setup that ``DynamicDML`` reports with blip features ``(1, X)``.

    A blip coefficient is reported when its feature is the intercept or a column of ``X``, and
    every static contrast is reported.  ``DynamicDML`` has no counterfactual mean, so no
    ``ey_*`` estimand is reported.
    """
    out = []
    for estimand in laws.setup(setup).estimands:
        match = re.fullmatch(r"blip\[\w+:[\w.]+:(\w+)\]", estimand)
        if (
            match and (match.group(1) == "intercept" or match.group(1) in features)
        ) or estimand.startswith("ate_regimen["):
            out.append(estimand)
    return tuple(out)


#: The cells the comparator's paired rows report.
REFERENCE_SCENARIOS: dict[str, tuple[str, ...]] = {
    scenario: comparator_estimands(SETUP_OF[scenario]) for scenario in COMPARATOR_PAIRED
}
#: The cells of the descriptive Markov table.
MARKOV_ESTIMANDS: dict[str, tuple[str, ...]] = {
    scenario: comparator_estimands(SETUP_OF[scenario], MARKOV_FEATURES.get(scenario, ()))
    for scenario in COMPARATOR_MARKOV
}


def _vector_text(values: Any) -> str:
    return ";".join(repr(float(value)) for value in values)


def comparator_tasks() -> pd.DataFrame:
    """The task table the comparator runner reads, one row per variant and estimand.

    It travels to the container as ``truth.csv``: the driver hands the runner the samples, this
    table and the output path.
    """
    rows = []
    variants = (("paired", REFERENCE_SCENARIOS), ("markov", MARKOV_ESTIMANDS))
    for variant, cells in variants:
        for scenario, estimands in cells.items():
            name = SETUP_OF[scenario]
            chosen = laws.setup(name)
            kind = treatment_kind(name)
            if variant == "paired":
                order, features = INTERACTIONS.get(name, DEFAULT_INTERACTIONS), ()
            else:
                order, features = MARKOV_ORDERS[kind], MARKOV_FEATURES.get(scenario, ())
            for estimand in estimands:
                match = re.fullmatch(r"ate_regimen\[(\w+) vs (\w+)\]", estimand)
                t1 = t0 = ""
                if match:
                    t1 = _vector_text(chosen.regimens[match.group(1)])
                    t0 = _vector_text(chosen.regimens[match.group(2)])
                rows.append(
                    {
                        "scenario": scenario,
                        "variant": variant,
                        "estimand": estimand,
                        "truth": laws.truth(name, estimand),
                        "kind": kind,
                        "nodes": chosen.law.n_nodes,
                        "order": order,
                        "features": ";".join(features),
                        "t1": t1,
                        "t0": t0,
                    }
                )
    return pd.DataFrame(rows, columns=list(TASK_COLUMNS))


PENDING_STUDY = StudyRecord(
    name="structural nested mean model",
    slug="structural-nested-mean",
    artifacts=ROOT / "tests" / "canonical" / "structural_nested_mean_study",
    document="docs/technical-reference/method-evidence/structural-nested-mean.md",
    anchor="structural-nested-mean-model",
    scenarios=dict(SCENARIOS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners=dict(SEED_OWNERS),
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    pending_reference=REFERENCE,
    pending_reason="the record before the comparator is registered; the registry tests read it",
    comparison="matching implementation",
    calibration_efficiency_ratio=False,
    modules=(
        "tests/studies/canonical_structural_nested_mean.py",
        "tests/studies/structural_nested_mean_properties.py",
        "tests/studies/structural_nested_mean_laws.py",
        "tests/discrete_law_structural.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_structural_nested_mean",
    properties_module="tests.studies.structural_nested_mean_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
    failure_bound={IMPLEMENTATION: FAILURE_BOUND, REFERENCE: FAILURE_BOUND},
)

#: The registered record.  The paired rows cover :data:`REFERENCE_SCENARIOS`, and the Markov rows
#: go to the three descriptive artefacts of :func:`reference_artifacts`.  ``exactness.csv`` is the
#: gated table of :func:`extra_artifacts`.
COMPARATOR_STUDY = dataclasses.replace(
    PENDING_STUDY,
    reference=REFERENCE,
    pending_reference=None,
    pending_reason="",
    reference_scenarios=dict(REFERENCE_SCENARIOS),
    extra_artifacts=(MARKOV_REPLICATES, MARKOV_SUMMARY, MARKOV_FAILURES, EXACTNESS_FILE),
)
STUDY = COMPARATOR_STUDY


def _booster_text(kind: str) -> str:
    settings = ", ".join(f"{key}={value!r}" for key, value in BOOSTER_SETTINGS.items())
    return f"{kind}({settings})"


def _support_text(declared: SupportDeclaration) -> dict[str, Any]:
    return {
        "treatment": list(declared.treatment),
        "weighting": declared.weighting,
        "dependence": declared.dependence,
        "covariate_features": declared.covariate_features,
        "regimens": declared.regimens,
        "learners": {role: cls.__name__ for role, cls in declared.learners.items()},
    }


CONFIGURATION = {
    "estimator": "StructuralNestedMean",
    "outcome_learner": {
        "GLM setups": f"GLM(family='gaussian', interactions=order); order {DEFAULT_INTERACTIONS}",
        "GLM orders": dict(INTERACTIONS),
        "N": _booster_text("HistGradientBoostingRegressor"),
    },
    "treatment_learner": {
        "GLM setups": "GLM(interactions=order), the outcome order unless listed",
        "GLM orders": dict(TREATMENT_INTERACTIONS),
        "N": _booster_text("HistGradientBoostingClassifier"),
    },
    "n_folds": {"default": DEFAULT_FOLDS, **FOLDS},
    "model_check": {
        "interactions": CHECK_INTERACTIONS,
        "n_splits": CHECK_SPLITS,
        "cells": {name: list(cell) for name, cell in CHECK_CELLS.items()},
    },
    "ladder": {setup: list(ladder_sizes(setup)) for setup in LADDER_ESTIMANDS},
    "calibration_n": CALIBRATION_N,
    "support_keys": {name: _support_text(declared) for name, declared in SUPPORT_KEYS.items()},
    "expected_red": list(EXPECTED_RED),
    "reference": (
        "none yet; the EconML 0.17.0 DynamicDML comparator is pending"
        if STUDY.reference is None
        else "EconML 0.17.0 DynamicDML (tests/canonical/econml_dynamic_dml/run_study.py)"
    ),
    "comparator": {
        "paired": {scenario: list(names) for scenario, names in REFERENCE_SCENARIOS.items()},
        "markov": {scenario: list(names) for scenario, names in MARKOV_ESTIMANDS.items()},
        "markov_features": {scenario: list(names) for scenario, names in MARKOV_FEATURES.items()},
        "markov_orders": dict(MARKOV_ORDERS),
        "fold_seed": COMPARATOR_FOLD_SEED,
        "markov_margin": MARKOV_MARGIN,
    },
}

# --------------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of ``scenario``'s law, with its exact truths."""
    name = SETUP_OF[scenario]
    return laws.sample(name, n, seed), laws.truths(name)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    rows = []
    for name in SCENARIOS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": IMPLEMENTATION,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(estimate.psi),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit(SETUP_OF[scenario], frame)
    require_status(result, SCENARIOS[scenario])
    return result_rows(result, truth, scenario, replicate)


class FoldMismatch(AssertionError):
    """The folds the comparator sample carries are not the subject fit's folds.

    An ``AssertionError`` stops the run: a paired row fitted on other folds would read as a
    numerical disagreement.
    """


def comparator_sample(scenario: str, replicate: int, frame: pd.DataFrame) -> pd.DataFrame:
    """The comparator's input for one replicate: the draw, its keys and the subject's folds.

    A scenario the comparator does not fit gets an empty frame.  The folds are drawn as
    :func:`fit` draws them, from :data:`COMPARATOR_FOLD_SEED`, so the sample needs no fit.
    """
    if scenario not in COMPARATOR_SCENARIOS:
        return pd.DataFrame()
    name = SETUP_OF[scenario]
    folds = random_partition(
        len(frame), FOLDS.get(name, DEFAULT_FOLDS), seed=COMPARATOR_FOLD_SEED
    ).assignment
    sample = frame.copy()
    sample.insert(0, "scenario", scenario)
    sample.insert(1, "replicate", replicate)
    sample["fold"] = np.asarray(folds, dtype=int)
    return sample


def _comparator_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """A failed subject fit's value: the sample alone, so the comparator still fits it."""
    scenario, replicate, _ = key
    return comparator_sample(scenario, replicate, drawn[0]), []


def _paired_replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample = comparator_sample(scenario, replicate, frame)
    result = fit(SETUP_OF[scenario], frame, random_state=COMPARATOR_FOLD_SEED)
    require_status(result, SCENARIOS[scenario])
    if not sample.empty and not np.array_equal(
        sample["fold"].to_numpy(), np.asarray(result.folds.assignment)
    ):
        raise FoldMismatch(
            f"{scenario} replicate {replicate}: the sample's folds are not the fit's"
        )
    return sample, result_rows(result, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    keys = [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS]
    if STUDY.reference is None:
        outcomes = replication.fit_primary(
            keys,
            _replicate,
            draw=_draw,
            n_jobs=n_jobs,
            implementation=IMPLEMENTATION,
            rows_at=None,
            seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
        )
        estimates = pd.DataFrame([row for rows in outcomes.kept for row in rows])
        return replication.result(outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)])
    outcomes = replication.fit_primary(
        keys,
        _paired_replicate,
        draw=_draw,
        reference_input=_comparator_input,
        n_jobs=n_jobs,
        implementation=IMPLEMENTATION,
        rows_at=1,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
        spill_samples_at=0,
    )
    samples = replication.assembled_samples(outcomes, at=0)
    estimates = pd.DataFrame([row for _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes,
        estimates.loc[:, list(REPLICATE_COLUMNS)],
        samples=samples,
        truths=comparator_tasks(),
    )


# ------------------------------------------------------------------- the descriptive table


def _sibling(path: Path, suffix: str) -> Path:
    """``<stem>.<suffix>.csv`` beside ``path``: where the runner writes its Markov files."""
    return path.with_name(f"{path.stem}.{suffix}.csv")


def markov_summary(rows: pd.DataFrame, failed: pd.DataFrame) -> pd.DataFrame:
    """One row per Markov cell: the fit's mean, spread and coverage beside the computed limit.

    ``limit`` is :func:`~tests.studies.structural_nested_mean_laws.dynamic_dml_limit` under the
    Markov state.  ``distance_low`` and ``distance_high`` bound the 99% interval of the mean's
    distance from the limit in empirical SDs.  ``agrees_with_limit`` is true when that interval
    lies inside :data:`MARKOV_MARGIN`.  ``coverage`` reads the truth and ``limit_coverage`` the
    limit.  The table describes the declared differences and gates nothing.
    """
    out = []
    lost = failed["scenarios"].astype(str).value_counts() if not failed.empty else pd.Series()
    for scenario, estimands in MARKOV_ESTIMANDS.items():
        name = SETUP_OF[scenario]
        features = MARKOV_FEATURES.get(scenario, ())
        for estimand in estimands:
            cell = rows.loc[(rows["scenario"] == scenario) & (rows["estimand"] == estimand)]
            estimate = cell["estimate"].to_numpy(dtype=float)
            count = len(estimate)
            truth = laws.truth(name, estimand)
            limit = laws.dynamic_dml_limit(name, estimand, state="markov", features=features)
            spread = float(np.std(estimate, ddof=1)) if count > 1 else math.nan
            mean = float(np.mean(estimate)) if count else math.nan
            distance = (mean - limit) / spread if spread > 0 else math.nan
            half = Z_99 / math.sqrt(count) if count else math.nan
            low, high = (
                cell["ci_lower"].to_numpy(dtype=float),
                cell["ci_upper"].to_numpy(dtype=float),
            )
            mean_se = float(cell["std_error"].mean()) if count else math.nan
            out.append(
                {
                    "scenario": scenario,
                    "estimand": estimand,
                    "replicates": count,
                    "failed": int(lost.get(scenario, 0)),
                    "truth": truth,
                    "limit": limit,
                    "gap": limit - truth,
                    "mean": mean,
                    "sd": spread,
                    "mean_std_error": mean_se,
                    "se_ratio": mean_se / spread if spread > 0 else math.nan,
                    "coverage": float(np.mean((low <= truth) & (truth <= high))),
                    "limit_coverage": float(np.mean((low <= limit) & (limit <= high))),
                    "distance_low": distance - half,
                    "distance_high": distance + half,
                    "agrees_with_limit": bool(
                        distance - half >= -MARKOV_MARGIN and distance + half <= MARKOV_MARGIN
                    ),
                }
            )
    return pd.DataFrame(out)


#: The two-sided 99% normal quantile.
Z_99 = 2.5758293035489004


def reference_artifacts(
    *,
    reference: Any,
    here: Path,
    samples: Path,
    truths_path: Path,
    reference_results: Path,
    output: Path,
    cores: int,
) -> dict[str, pd.DataFrame]:
    """The Markov rows the comparator run wrote beside its paired rows, and their table.

    The runner fits both variants in one container run, so this hook runs no process.  It
    reads ``<reference_results stem>.markov.csv`` and ``.markov-failures.csv``.
    """
    del reference, here, samples, truths_path, output, cores
    rows = pd.read_csv(_sibling(reference_results, "markov"))
    rows = rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["scenario", "replicate", "estimand"], ignore_index=True
    )
    failures_path = _sibling(reference_results, "markov-failures")
    failed = (
        pd.read_csv(failures_path, dtype={"scenarios": str, "replicates": str})
        if failures_path.exists()
        else pd.DataFrame(columns=list(MARKOV_FAILURE_COLUMNS))
    )
    return {
        MARKOV_REPLICATES: rows,
        MARKOV_SUMMARY: markov_summary(rows, failed),
        MARKOV_FAILURES: failed.loc[:, list(MARKOV_FAILURE_COLUMNS)],
    }


#: The columns of the runner's failure records.
MARKOV_FAILURE_COLUMNS = (
    "group_id",
    "scenarios",
    "replicates",
    "attempt",
    "class",
    "stage",
    "call",
    "message",
    "seed",
)


EXACTNESS_AUDIT = "paired estimate exactness"
MARKOV_AUDIT = "Markov-state DynamicDML table (descriptive)"


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``exactness.csv``: one row per paired cell, gated on the estimate gap only.

    Each paired cell compares the subject and ``DynamicDML`` on the same rows, folds and
    ``GLM`` design, so their estimates agree to numerical precision.  A row pairs the two sides
    on ``(scenario, replicate, estimand)``.  ``max_estimate_gap`` is the largest absolute
    difference of the paired estimates, and the cell passes when it is below :data:`EXACTNESS_GATE`
    and at least one pair exists.  The standard errors differ by design, by the factor
    :math:`\\sqrt{(n - 1) / n}` on a scalar node and through the transposed blocks on ``k``.
    The table publishes their ratio, reference over subject, and does not gate it.
    """
    keys = ["scenario", "replicate", "estimand"]
    columns = [*keys, "estimate", "std_error"]
    sides = {
        name: rows.loc[rows["implementation"] == name, columns]
        for name in (IMPLEMENTATION, REFERENCE)
    }
    paired = sides[IMPLEMENTATION].merge(sides[REFERENCE], on=keys, suffixes=("_s", "_r"))
    paired["estimate_gap"] = (paired["estimate_s"] - paired["estimate_r"]).abs()
    paired["se_ratio"] = paired["std_error_r"] / paired["std_error_s"]
    summary = (
        paired.groupby(["scenario", "estimand"])
        .agg(
            paired_replicates=("replicate", "size"),
            max_estimate_gap=("estimate_gap", "max"),
            mean_se_ratio=("se_ratio", "mean"),
            min_se_ratio=("se_ratio", "min"),
            max_se_ratio=("se_ratio", "max"),
        )
        .reset_index()
    )
    cells = pd.DataFrame(sorted(STUDY.paired_cells()), columns=["scenario", "estimand"])
    table = cells.merge(summary, on=["scenario", "estimand"], how="left")
    table["paired_replicates"] = table["paired_replicates"].fillna(0).astype(int)
    table["gate"] = EXACTNESS_GATE
    table["passed"] = (table["paired_replicates"] > 0) & (
        table["max_estimate_gap"] < EXACTNESS_GATE
    )
    return {EXACTNESS_FILE: table.loc[:, list(EXACTNESS_COLUMNS)]}


#: The columns of ``exactness.csv``.
EXACTNESS_COLUMNS = (
    "scenario",
    "estimand",
    "paired_replicates",
    "max_estimate_gap",
    "mean_se_ratio",
    "min_se_ratio",
    "max_se_ratio",
    "gate",
    "passed",
)


def scientific_failures(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The paired cells whose largest estimate gap reached :data:`EXACTNESS_GATE`."""
    table = artifacts[EXACTNESS_FILE]
    return {EXACTNESS_AUDIT: table.loc[~table["passed"].astype(bool)]}


def _lost_pairs(failures: pd.DataFrame) -> int:
    """The (scenario, replicate) paired fits that either side lost, counted once."""
    lost = evidence_failures.expand(failures, STUDY)
    lost = lost.loc[lost["scenario"].isin(COMPARATOR_PAIRED)]
    return len(lost[["scenario", "replicate"]].drop_duplicates())


def account_failures(failures: pd.DataFrame, artifacts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """A paired replication that either side lost leaves its cells unpaired, so it counts.

    The Markov table gates nothing.  Its own lost fits are counted in its ``failed`` column and
    kept in :data:`MARKOV_FAILURES`.
    """
    return evidence_failures.audits(
        {**scientific_failures(artifacts), MARKOV_AUDIT: pd.DataFrame()},
        missing={EXACTNESS_AUDIT: _lost_pairs(failures)},
    )


__all__ = [
    "COMPARATOR_STUDY",
    "STUDY",
    "draw_and_fit",
    "extra_artifacts",
    "fit",
    "reference_artifacts",
    "scientific_failures",
]
