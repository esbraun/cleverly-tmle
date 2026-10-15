r"""Registered evidence for the structural nested mean estimator (``structural-nested-mean``, X44).

The subject is :class:`~cleverly.longitudinal.StructuralNestedMean` with declared
:class:`~cleverly.learners.GLM` nuisances (and a shallow regularized booster on the nonlinear law
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
``n``     nonlinear nuisances, a cross-fitted booster
``s2w``   known weights
``s2c``   clusters of ten
========  ======================================================================================

**Negative controls.**  ``LU`` is law ``L`` fitted with :math:`\phi_2 = 1`, and ``VB`` is law
``P`` fitted with a baseline-only :math:`\phi_2`.  Their regimen contrasts miss the regimen
truth.  They are property cells (family ``blip_feature_control``), not primary scenarios: the
replication check refuses a primary cell with coverage below 50% as a broken join.  The
roadmap's "truth row owned by the structural label" therefore reads as a red property row with
the owner ``X44-negative-control``.

**Red-cell reading** (declared before any run).  The study publishes under ``reporting``.  The
declared red set is :data:`EXPECTED_RED`:

* every cell of the family ``blip_feature_control`` (``EXPECTED_RED_PROPERTIES``): the
  limit of the undeclared fit misses the regimen truth, so the calibration rule must fail.  The
  design test computes each limit by the g-formula and the chance that the cell fails;
* the model-check size cell of the sparse-cell law ``SP`` (``EXPECTED_RED_PROPERTIES``): at
  n = 500 with treatment logits tripled, the pilot recorded a singular-information ``DataError``
  in 19% of the draws (above the failure bound) and rejected 14.5% of the others (above the 0.10
  ceiling).  Both are finite-sample limits of the unpenalized ``GLM`` nuisances and not defects
  of the statistic.  The cell is declared red, owner ``X44-sparse-model-check``, and the page
  says so beside the cell.

A red row outside :data:`EXPECTED_RED` is a defect suspect.  It is a finite-sample red only when
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
#: The three sizes of the ladder.  Every rung is a positive cell (pilot: ``root_n_pilot`` in
#: ``tests/unit/test_structural_nested_mean_design.py``).
LADDER = (500, 2_000, 8_000)
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
#: three on ``S2`` fails on a separated cell in 9.7% of the draws at n = 500.
INTERACTIONS: dict[str, int] = {
    "S2": 2,
    "H": 2,
    "P": 2,
    "L": 2,
    "K": 2,
    "S2W": 2,
    "S2C": 2,
    "C": 1,
}
DEFAULT_INTERACTIONS = 3
#: The outer folds of each setup; ``10`` (the shipped default) unless listed.
DEFAULT_FOLDS = 10
FOLDS: dict[str, int] = {"N": 5}


def booster_classifier() -> HistGradientBoostingClassifier:
    """The shallow regularized booster of ``N``'s treatment mechanisms."""
    return HistGradientBoostingClassifier(
        max_depth=2,
        max_iter=50,
        learning_rate=0.1,
        l2_regularization=1.0,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=0,
    )


def booster_regressor() -> HistGradientBoostingRegressor:
    """The shallow regularized booster of ``N``'s outcome regressions."""
    return HistGradientBoostingRegressor(
        max_depth=2,
        max_iter=50,
        learning_rate=0.1,
        l2_regularization=1.0,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=0,
    )


#: The ``GLM`` interaction order of the model-check fits.  At n = 1,000 and two folds an order of
#: three fails on a separated follower cell in about 14% of the draws of ``S2``; an order of two
#: fails in none of 150.
CHECK_INTERACTIONS = 2


def learners(name: str, order: int | None = None) -> dict[str, Any]:
    """The outcome and treatment learners of one setup; ``order`` replaces its ``GLM`` order."""
    if laws.setup(name).nuisance == "booster":
        return {
            "outcome_learner": booster_regressor(),
            "treatment_learner": booster_classifier(),
        }
    order = INTERACTIONS.get(name, DEFAULT_INTERACTIONS) if order is None else order
    return {
        "outcome_learner": GLM(family="gaussian", interactions=order),
        "treatment_learner": GLM(interactions=order),
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
#: The ladder contrast of each laddered setup.
LADDER_ESTIMANDS: dict[str, str] = {
    "S2": "ate_regimen[r11 vs r00]",
    "H": "ate_regimen[r11 vs r00]",
    "K": "ate_regimen[r12 vs r00]",
    "C": "ate_regimen[dose_high vs dose0]",
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
LADDER_LABELS: tuple[str, ...] = tuple(
    label_of(setup, estimand) for setup, estimand in LADDER_ESTIMANDS.items()
)


def truth_of(label: str) -> float:
    """The exact truth of one calibration label."""
    setup, estimand = TARGETS[label]
    return laws.truth(setup, estimand)


#: The sizes of the model-check cells and their declared replicates, sized by the design test.
SIZE_N = 1_000
SPARSE_N = 500
SIZE_REPLICATES = 600
POWER_N = 2_000
POWER_REPLICATES = 200
#: The check cells: ``name -> (kind, setup, n, folds)``.
CHECK_CELLS: dict[str, tuple[str, str, int, int]] = {
    "s2_f2": ("size", "S2", SIZE_N, 2),
    "s2_f10": ("size", "S2", SIZE_N, 10),
    "h_f2": ("size", "H", SIZE_N, 2),
    "h_f10": ("size", "H", SIZE_N, 10),
    "sp_f2": ("size", "SP", SPARSE_N, 2),
    "lu_f2": ("power", "LU", POWER_N, 2),
    "lu_f10": ("power", "LU", POWER_N, 10),
}

#: Every published property cell, by family.  :mod:`tests.studies.structural_nested_mean_properties`
#: declares how each is drawn and read.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "root_n_and_efficiency": tuple(
        f"{label}__n_{size}" for label in LADDER_LABELS for size in LADDER
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
    "unclustered_se": tuple(
        f"{label_of('S2C', estimand)}__unclustered" for estimand in laws.setup("S2C").estimands
    ),
    "blip_feature_control": tuple(f"{label}__too_few_features" for label in CONTROL_LABELS),
}

#: The property rows declared red: every negative-control cell (the calibration rule must fail
#: there, and the design test shows that it does with chance at least 0.95) and the sparse-cell
#: size cell of the model check.
EXPECTED_RED_PROPERTIES: tuple[str, ...] = (
    *(f"blip_feature_control/{cell}" for cell in PROPERTY_CELLS["blip_feature_control"]),
    "warning_rate/sp_f2__model_check",
)
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

CONFIGURATION = {
    "estimator": "StructuralNestedMean",
    "outcome_learner": {
        "GLM setups": f"GLM(family='gaussian', interactions=order); order {DEFAULT_INTERACTIONS}",
        "GLM orders": dict(INTERACTIONS),
        "N": "HistGradientBoostingRegressor(max_depth=2, max_iter=50, l2=1)",
    },
    "treatment_learner": {
        "GLM setups": "GLM(interactions=order), the same orders",
        "N": "HistGradientBoostingClassifier(max_depth=2, max_iter=50, l2=1)",
    },
    "n_folds": {"default": DEFAULT_FOLDS, **FOLDS},
    "ladder": list(LADDER),
    "calibration_n": CALIBRATION_N,
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
