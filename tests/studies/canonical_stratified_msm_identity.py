"""Registered evidence for the identity-link MSM with baseline strata.

A stratified identity-link MSM fit solves one score block per stratum in one pooled
fluctuation, and each stratum's coefficients are the projection on the law given ``V = s``.
The shipped construction had no registered evidence.  The law is L1 of
:mod:`tests.studies.stratified_alternating_law` with the bounded outcome
``Y ~ Beta(24 Q, 24 (1 - Q))`` of the point-treatment MSM study, so ``E[Y | A, W, V]`` is L1's
``Q`` and every truth is exact.

The primary scenario pairs the nine stratum coefficients of ``MSM.linear(modifiers=("W",),
interaction=False)`` with pinned R ``tmle3`` ``Param_MSM`` at ``ed72f8a``, called once per
stratum subset by the shipped ``tmle3_msm`` construction (msm ``A + V``, ``V`` the
covariate ``W``, uniform weight, arm-indicator coefficients transformed to the ``(1, a, W)``
basis).  ``tmle3`` has no stratified MSM: ``Param_stratified`` cannot assemble an MSM curve, and a
literal partition needs none.

A 40-replication smoke run at n = 2,000, before the declaration, estimated the paired margin
utilization at most 0.13.

Declared before any run: the law, n = 2,000, R = 1,000 primary replications, the shared
``Margins()``, the learners, the seeds below, the property cells of
:mod:`tests.studies.stratified_msm_identity_properties`, and ``publication_policy="gated"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.msm import MSM
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_alternating_law as law
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.point_study_helpers import primary_rows

TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20268811
RESAMPLING_SEED = 20268812
SCENARIO = "stratified_msm_identity"
COVARIATES = ("W", "V")
G_BOUNDS = (0.01, 0.99)

#: The nine paired coefficients, stratum by stratum.
ESTIMANDS: tuple[str, ...] = tuple(name for s in law.STRATA for name in law.msm_names(s))
EFFICIENCY_SD = {name: law.beta_efficiency_sd(name) for name in ESTIMANDS}
TERM_KEYS = {"(intercept)": "intercept", "a": "a", "W": "w"}

#: ``identity_<scope>_<term>``: each coefficient of the calibration fits.
CALIBRATION_LABELS: tuple[str, ...] = tuple(
    f"identity_{scope}_{key}"
    for scope in ("marginal", *(f"v{stratum}" for stratum in law.STRATA))
    for key in TERM_KEYS.values()
)
CALIBRATION_KINDS = ("correctly_specified", "shrunken_se_control", "noise_control")


def label_name(label: str) -> str:
    """The package's name of the coefficient a calibration label reads."""
    _, scope, key = label.split("_")
    term = next(name for name, short in TERM_KEYS.items() if short == key)
    return f"msm[{term}]" + ("" if scope == "marginal" else f"[V={int(scope[1:])}]")


PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}" for label in CALIBRATION_LABELS for kind in CALIBRATION_KINDS
    ),
}

STUDY = StudyRecord(
    name="identity-link MSM with baseline strata",
    slug="canonical-stratified-msm-identity",
    artifacts=ROOT / "tests" / "canonical" / "tmle3_stratified_msm",
    document="docs/technical-reference/method-evidence/stratified-msm-identity.md",
    anchor="stratified-identity-link-msm",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-stratified-msm",
    reference="tmle3-msm-stratified",
    modules=(
        "tests/studies/canonical_stratified_msm_identity.py",
        "tests/studies/stratified_msm_identity_properties.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/discrete_law.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_stratified_msm_identity",
    properties_module="tests.studies.stratified_msm_identity_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_SD,
    publication_policy="gated",
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": (
        "tmle3 Param_MSM with Gaussian identity-link projection, once per stratum subset"
    ),
}

CONFIGURATION = {
    "law": (
        "L1 with a bounded outcome: V in {0, 1, 2}; W in {0, 1, 2} given V; "
        "g = expit(-0.3 + 0.5 W - 0.4 V); Y ~ Beta(24 Q, 24 (1 - Q)) with "
        "Q = expit(-0.8 + 1.0 A + 0.6 W + 0.9 V)"
    ),
    "working_model": "MSM.linear(modifiers=('W',), interaction=False): (1, a, W), h = 1",
    "cross_fit": False,
    "strata": ["V"],
    "covariates": list(COVARIATES),
    "g_bounds": list(G_BOUNDS),
    "simultaneous_intervals": False,
    "outcome_model": "quasibinomial GLM on (A, W, V)",
    "treatment_model": "LogisticRegression(C=1e6) on (W, V)",
    "reference_learners": "sl3 Lrnr_glm for A and Y on (W, W^2) inside each stratum",
    "declared_difference": (
        "tmle3 fits its nuisances inside each stratum subset with Lrnr_glm; the cleverly "
        "fit pools the nuisances over the strata and solves one block per stratum"
    ),
}


def logistic() -> LogisticRegression:
    """The unpenalized main-terms logistic regression of the treatment."""
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


def model() -> MSM:
    """The declared working model ``(1, a, W)`` with uniform weights."""
    return MSM.linear(modifiers=("W",), interaction=False)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One bounded-outcome sample of L1, with every identity-MSM truth."""
    if scenario != SCENARIO:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.beta_sample(n, seed), law.truths("identity")


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, *, simultaneous: bool = False) -> Any:
    """The in-sample stratified identity-MSM fit."""
    return (
        TMLE(
            msm=model(),
            outcome_learner=QuasiBinomialGLM(),
            treatment_learner=logistic(),
            cross_fit=False,
            simultaneous=simultaneous,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=list(COVARIATES), strata=["V"])
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
