r"""Registered evidence for the structural nested mean estimator (``structural-nested-mean``, X44).

The subject is :class:`~cleverly.longitudinal.StructuralNestedMean` with declared
:class:`~cleverly.learners.GLM` nuisances (and an early-stopped booster on the nonlinear law
``N``).  The laws are in :mod:`tests.studies.structural_nested_mean_laws` and the property
families in :mod:`tests.studies.structural_nested_mean_properties`.

This study is Python only.  The EconML ``DynamicDML`` comparator is declared and pending
(``pending_reference``): the container phase is a later change, and the record does not depend
on it.  The fast-tier parity fixture of ``tests/canonical/econml_dynamic_dml`` already checks the
point estimates on one panel.

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

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleverly.learners import GLM
from cleverly.longitudinal import StructuralNestedMean
from tests.parallel import STUDY_JOBS
from tests.studies import structural_nested_mean_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SEED = 20261444
RESAMPLING_SEED = 2026144401

#: The size of every primary scenario and of every calibration cell.
PRIMARY_N = 2_000
CALIBRATION_N = PRIMARY_N
PRIMARY_REPLICATES = 1_000
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

#: The pending comparator, and the roadmap item that owns its runs.
PENDING_REFERENCE = "econml"
PENDING_REASON = (
    "the EconML 0.17.0 DynamicDML runs are a local container phase (tests/canonical/"
    "econml_dynamic_dml); a method merges once its Python studies pass and a later change "
    "registers the comparator runs"
)

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

STUDY = StudyRecord(
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
    pending_reference=PENDING_REFERENCE,
    pending_reason=PENDING_REASON,
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
    failure_bound={IMPLEMENTATION: FAILURE_BOUND, PENDING_REFERENCE: FAILURE_BOUND},
)


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
    "reference": "none yet; the EconML 0.17.0 DynamicDML comparator is pending",
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


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=IMPLEMENTATION,
        rows_at=None,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    estimates = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)])


__all__ = ["STUDY", "draw_and_fit", "fit"]
