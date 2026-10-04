"""Registered evidence for stratified incremental and MSM targeting.

A stratified incremental fit solves one outcome block and one treatment-mechanism block per
baseline stratum, and a stratified linked MSM fit solves one block per stratum at that stratum's
own coefficients.  The law is ``L1`` of :mod:`tests.studies.stratified_alternating_law`, which
is the baseline-strata law of :mod:`tests.studies.stratified_law`, and every truth is exact.

The primary scenario pairs the in-sample stratified incremental fit with pinned R ``npcausal``
``ipsi`` at ``56a5ac1``.  ``npcausal`` has no stratified form, so the runner calls ``ipsi`` once
per stratum subset and multiplier, and once per multiplier on every row for the marginal means.
``cleverly`` fits saturated nuisances over the ``(W, V)`` cells.  ``npcausal`` fits
``SL.glm.interaction`` (``Y ~ .^2``): over the ``W`` indicators inside a stratum it is saturated,
and over the ``W`` and ``V`` indicators on every row it has no ``A:W:V`` terms, so it is correct
for L1's main-terms outcome but not saturated.  The declared difference is
``npcausal``'s two-split cross-fitting (``nsplits = 2``; the single-split path selects no
training rows at this commit).

A 40-replication smoke run at n = 2,000, before the declaration, estimated the paired margin
utilization as the mean difference over the margin: at most 0.70 (the marginal
``ate_ipsi[odds x2 vs natural course]``), 0.07 to 0.64 for the stratum contrasts and at most
0.23 for the means.  The plan forecast about 0.7 in the smallest stratum, from the cross-fitting
difference scaling like ``1 / n_s``.  R stays at 1,600, the marginal study's budget.

Declared before any run: the laws, n = 2,000, R = 1,600 primary replications, the shared
``Margins()``, the learners, the seeds below, the property cells of
:mod:`tests.studies.stratified_incremental_msm_properties`, and ``publication_policy="gated"``.
A red cell is not repaired with a budget, a margin, a law, a learner, a size or a seed.  A
stratum, block, coefficient, embedding or reduction bug is a defect: it is fixed, given a
witness, and the study is regenerated once.  A finite-sample red with no defect publishes under
``reporting`` after a re-run at the same seeds.

The declared run at ``a1b01d5`` published two red cells under ``gated``:
``interval_calibration/logit_v1_a__correctly_specified`` (SE-ratio interval 0.927 to 1.010
against a floor of 0.93) and ``logit_v2_a__correctly_specified`` (0.782 to 0.939).
``tests/diagnostics/x8_logit_small_stratum`` read them as a finite-sample shortfall of the
logit-MSM slope at a stratum of about 400 rows.  The shipped unstratified fit on samples of that
size from the law given ``V = 2`` under-reports alike (0.896 against 0.867), and both arms are
calibrated at four times the size (0.997 and 1.006).  No defect was found, so the study moved to
``reporting`` and was re-run at the same seeds.  The owner is ``X8-logit-small-stratum``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly.estimators import TMLE
from cleverly.interventions import Incremental
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_alternating_law as law
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

NPCAUSAL_COMMIT = "56a5ac117a29258b67b94874be662a171b5131f7"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20268801
RESAMPLING_SEED = 20268802
SCENARIO = "stratified_incremental"
COVARIATES = ("W", "V")

#: The three tilts, in the order the fit declares them.  The first is the reference.
TILTS = tuple(Incremental(delta) for delta in law.DELTAS.values())

#: The twenty paired names: the five marginal ones, then each stratum's five.
ESTIMANDS: tuple[str, ...] = law.all_ipsi_names()
EFFICIENCY_SD = {name: law.efficiency_sd(name) for name in ESTIMANDS}

# ------------------------------------------------------------------------------ cells

#: The short key of each incremental name in a property cell.
IPSI_KEYS = {
    "ey_ipsi[natural course]": "ey_natural",
    "ey_ipsi[odds x2]": "ey_x2",
    "ey_ipsi[odds x0.5]": "ey_x05",
    "ate_ipsi[odds x2 vs natural course]": "ate_x2",
    "ate_ipsi[odds x0.5 vs natural course]": "ate_x05",
}
#: The short key of each MSM term.
TERM_KEYS = {"(intercept)": "intercept", "a": "a", "W": "w"}

#: ``v<s>_<key>``: each stratum incremental parameter of the calibration fits.
IPSI_LABELS: tuple[str, ...] = tuple(
    f"v{stratum}_{key}" for stratum in law.STRATA for key in IPSI_KEYS.values()
)
#: ``logit_<scope>_<term>``: each logit-MSM coefficient, marginal and stratum.
MSM_LABELS: tuple[str, ...] = tuple(
    f"logit_{scope}_{key}"
    for scope in ("marginal", *(f"v{stratum}" for stratum in law.STRATA))
    for key in TERM_KEYS.values()
)
#: ``continuous_<link>_<scope>_<term>``: each continuous-dose coefficient.  These are
#: ``outcome_correct`` robustness cells, not calibration cells: the treatment mechanism is the
#: package's binned density, which is not the law's density, so the reported standard error
#: is not the estimator's.  A 400-replication smoke run before the declaration measured SE
#: ratios of 0.63 to 1.52 and standardized biases of at most 0.13.
CONTINUOUS_TERMS = {"(intercept)": "intercept", "a": "a"}
CONTINUOUS_LABELS: tuple[str, ...] = tuple(
    f"continuous_{link}_{scope}_{key}"
    for link in ("identity", "logit")
    for scope in ("marginal", *(f"v{stratum}" for stratum in law.STRATA))
    for key in CONTINUOUS_TERMS.values()
)
#: The necessity families read the two outer strata, as the strata study does.
NECESSITY_STRATA = (0, 2)
NECESSITY_LABELS: tuple[str, ...] = tuple(f"v{stratum}_ey_x2" for stratum in NECESSITY_STRATA)
NECESSITY_ARMS = ("stratified", "marginal_fluctuation")
#: The MSM targeting-necessity family reads strata 0 and 1.  The untargeted projection of a
#: ``Q`` that omits ``V`` is 1.8 and 1.1 SD off the truth there and 0.5 SD in stratum 2, so
#: stratum 2 does not resolve the pair (``tests/unit/test_stratified_alternating_design.py``).
TARGETING_LABELS: tuple[str, ...] = ("v0_a", "v1_a")
TARGETING_ARMS = ("targeted", "untargeted")
#: The robustness family fits the logit MSM ``(1, a)``, :data:`law.ARM_TERMS`.  Inside a
#: stratum the model ``(1, a, W)`` contains ``Q``, so a fit with both nuisances wrong stays
#: within 0.5 SD of the truth and its control could not fail.  Without ``W`` the both-wrong
#: limit is the confounded stratum slope (the design test computes each limit).
DOUBLE_ROBUST_LABELS: tuple[str, ...] = tuple(f"v{stratum}_a_arm" for stratum in law.STRATA)
DOUBLE_ROBUST_CONFIGURATIONS = ("both_correct", "outcome_correct", "treatment_correct")
BOTH_WRONG_LABELS: tuple[str, ...] = DOUBLE_ROBUST_LABELS
#: ``natural_<scope>``: the cross-fitted natural-course mean of L1 with a missing outcome,
#: marginal and stratum.
NATURAL_COURSE_LABELS: tuple[str, ...] = tuple(
    f"natural_{scope}" for scope in ("marginal", *(f"v{stratum}" for stratum in law.STRATA))
)
JOINT_LABELS = ("ipsi_strata",)
CALIBRATION_KINDS = ("correctly_specified", "shrunken_se_control", "noise_control")


def label_name(label: str) -> str:
    """The package's name of the parameter an incremental property label reads."""
    stratum, key = label.split("_", 1)
    stem = next(name for name, short in IPSI_KEYS.items() if short == key)
    return f"{stem}[V={int(stratum[1:])}]"


def natural_label_name(label: str) -> str:
    """The package's name of the natural-course mean a label reads."""
    scope = label.split("_", 1)[1]
    return "ey_obs" if scope == "marginal" else f"ey_obs[V={int(scope[1:])}]"


def msm_label_name(label: str) -> str:
    """The package's name of the coefficient an MSM or continuous label reads."""
    parts = label.split("_")
    scope, key = parts[-2], parts[-1]
    keys = CONTINUOUS_TERMS if label.startswith("continuous") else TERM_KEYS
    term = next(name for name, short in keys.items() if short == key)
    suffix = "" if scope == "marginal" else f"[V={int(scope[1:])}]"
    return f"msm[{term}]{suffix}"


PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in (*IPSI_LABELS, *MSM_LABELS, *NATURAL_COURSE_LABELS)
        for kind in CALIBRATION_KINDS
    ),
    "simultaneous_coverage": tuple(
        f"{label}__{kind}"
        for label in JOINT_LABELS
        for kind in ("simultaneous_band", "pointwise_joint_control")
    ),
    "stratum_targeting_necessity": tuple(
        f"{label}__{arm}" for label in NECESSITY_LABELS for arm in NECESSITY_ARMS
    ),
    "double_robustness": (
        *(
            f"{label}__{configuration}"
            for label in DOUBLE_ROBUST_LABELS
            for configuration in DOUBLE_ROBUST_CONFIGURATIONS
        ),
        *(f"{label}__both_wrong" for label in BOTH_WRONG_LABELS),
        *(f"{label}__outcome_correct" for label in CONTINUOUS_LABELS),
    ),
    "targeting_necessity": tuple(
        f"{label}__{arm}" for label in TARGETING_LABELS for arm in TARGETING_ARMS
    ),
}


STUDY = StudyRecord(
    name="stratified incremental and MSM targeting",
    slug="canonical-stratified-incremental-msm",
    artifacts=ROOT / "tests" / "canonical" / "npcausal_stratified_incremental",
    document="docs/technical-reference/method-evidence/stratified-incremental-msm.md",
    anchor="stratified-incremental-msm",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-stratified-incremental",
    reference="npcausal-stratified",
    modules=(
        "tests/studies/canonical_stratified_incremental_msm.py",
        "tests/studies/stratified_incremental_msm_properties.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/discrete_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_stratified_incremental_msm",
    properties_module="tests.studies.stratified_incremental_msm_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_SD,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "npcausal_commit": NPCAUSAL_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "point-treatment incremental odds curve, per stratum subset",
    "reference_scope": "point estimates and efficient-influence-curve inference",
    "reference_nuisances": (
        "two-fold cross-fitted SL.glm.interaction (Y ~ .^2) over the W indicators inside a "
        "stratum, which is saturated there, and over the W and V indicators on every row for "
        "the marginal means, which has no A:W:V terms and is correct but not saturated"
    ),
}

CONFIGURATION = {
    "law": (
        "L1: V in {0, 1, 2} with P(V) = (0.5, 0.3, 0.2); W in {0, 1, 2} given V; "
        "g = expit(-0.3 + 0.5 W - 0.4 V); Q = expit(-0.8 + 1.0 A + 0.6 W + 0.9 V); binary Y"
    ),
    "cross_fit": False,
    "strata": ["V"],
    "covariates": list(COVARIATES),
    "deltas": dict(law.DELTAS),
    "simultaneous_intervals": False,
    "outcome_model": "saturated: one probability per (A, W, V) cell",
    "treatment_model": "saturated: one probability per (W, V) cell",
    "declared_difference": (
        "npcausal ipsi cross-fits its nuisances with nsplits = 2; the cleverly fit is in "
        "sample.  ipsi is called once per stratum subset for the stratum names, and once on "
        "every row for the five marginal names of the stratified fit"
    ),
}


class CellFrequency(BaseEstimator, ClassifierMixin):
    """The saturated classifier: the weighted share of ``y = 1`` within each design row.

    On a finite-support law it is the unpenalized maximum-likelihood fit of a model with one
    parameter per cell, which is what ``SL.glm.interaction`` spans inside a stratum.  A design
    row the fit never saw falls back to the overall share.
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> CellFrequency:
        matrix = np.round(np.asarray(design, dtype=float), 9)
        y = np.asarray(target, dtype=float).reshape(-1)
        weights = (
            np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        keys, inverse = np.unique(matrix, axis=0, return_inverse=True)
        totals = np.bincount(inverse, weights=weights * y, minlength=len(keys))
        sizes = np.bincount(inverse, weights=weights, minlength=len(keys))
        self.keys_ = {tuple(key): index for index, key in enumerate(keys)}
        self.shares_ = totals / np.where(sizes > 0, sizes, 1.0)
        self.default_ = float(np.average(y, weights=weights))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        matrix = np.round(np.asarray(design, dtype=float), 9)
        p = np.array(
            [
                self.shares_[self.keys_[tuple(row)]] if tuple(row) in self.keys_ else self.default_
                for row in matrix
            ]
        )
        return np.column_stack([1.0 - p, p])


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of L1 from an explicit seed, with every incremental truth."""
    if scenario != SCENARIO:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.sample(n, seed), law.truths()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    outcome_learner: Any = None,
    treatment_learner: Any = None,
    simultaneous: bool = False,
    strata: bool = True,
) -> Any:
    """The in-sample stratified incremental fit, or a declared variant of it."""
    return (
        TMLE(
            incremental=list(TILTS),
            outcome_learner=CellFrequency() if outcome_learner is None else outcome_learner,
            treatment_learner=CellFrequency() if treatment_learner is None else treatment_learner,
            cross_fit=False,
            simultaneous=simultaneous,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(COVARIATES),
            **({"strata": ["V"]} if strata else {}),
        )
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one replication."""
    return primary_rows(
        result=fit_cleverly(frame),
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": reference[name]}
        for name in ESTIMANDS
    ]
    return sample, truth_rows, cleverly_rows(frame, reference, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw every replication, fit the subject, and return the rows the runner reads."""
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]
