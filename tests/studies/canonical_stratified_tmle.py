"""Registered evidence for ordinary point-treatment TMLE with baseline strata.

A stratum parameter is the marginal point-treatment parameter of the law inside one of a
fixed number of baseline strata.  ``TMLE(...).fit(..., strata=["V"])`` fluctuates once, with one
disjoint score block ``I(S=s) H_s / P_n(S=s)`` per stratum, and embeds each stratum curve as
``I(S=s) D_s / P_n(S=s)``.  The law is :mod:`tests.studies.stratified_law`, which gives every
truth and every efficient influence function exactly.

The primary scenario pairs the in-sample fit with pinned R ``tmle3`` at ``ed72f8a``, through
``tmle_stratified(..., base_estimate = FALSE)`` in two stages: ``tmle_TSM_all()`` for the six
stratum arm means and ``tmle_ATE(1, 0)`` for the three stratum contrasts.  The ``cleverly`` fit
requests ``("ey", "ate")``, so its two arm columns carry both.  Declared differences and
exclusions are on the study page and in :data:`CONFIGURATION`.

Declared before any run: the law, n = 2,000, R = 1,000 primary
replications, the shared ``Margins()``, the main-terms logistic learners on ``(A, W, V)`` and
``(W, V)``, ``g_bounds=(0.01, 0.99)``, the seeds below, the property cells of
:mod:`tests.studies.stratified_tmle_properties`, and ``publication_policy="gated"``.  A red
cell is not repaired with a budget, a margin, a law, a learner, a size or a seed.  A stratum
alignment or embedding bug in ``cleverly`` is a defect: it is fixed, given a witness, and the
study is regenerated.  A comparator defect is worked around in the runner and recorded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_law as law
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20263611
RESAMPLING_SEED = 20263612
SCENARIO = "stratified_binary"
G_BOUNDS = (0.01, 0.99)
COVARIATES = ("W", "V")

#: The nine paired parameters: both arm means and their contrast, stratum by stratum.
ESTIMANDS: tuple[str, ...] = tuple(
    law.stratum_name(stem, stratum) for stratum in law.STRATA for stem in ("ey[1]", "ey[0]", "ate")
)

#: The exact efficiency-bound SD of every paired parameter.
EFFICIENCY_SD = {name: law.EFFICIENCY_SD[name] for name in ESTIMANDS}

# ------------------------------------------------------------------------------ cells

#: The short key of each stem in a property cell name.
KEYS = {"ey[1]": "ey1", "ey[0]": "ey0", "ate": "ate", "att": "att", "atc": "atc", "par": "par"}

#: ``v<s>_<key>``: each stratum parameter of the in-sample calibration fit.
CALIBRATION_LABELS: tuple[str, ...] = tuple(
    f"v{stratum}_{key}" for stratum in law.STRATA for key in KEYS.values()
)
#: ``v<s>_ate_crossfit``: each stratum ATE of the default cross-fitted fit.
CROSSFIT_LABELS: tuple[str, ...] = tuple(f"v{stratum}_ate_crossfit" for stratum in law.STRATA)
#: ``v<s>_ate``: each stratum ATE of the robustness family.
ATE_LABELS: tuple[str, ...] = tuple(f"v{stratum}_ate" for stratum in law.STRATA)
#: The necessity family reads the two outer strata.  The omitted-``V`` bias of its control
#: nearly vanishes in the middle one, so a control there could not fail; the law's docstring
#: gives the numbers.
NECESSITY_STRATA = (0, 2)
NECESSITY_LABELS: tuple[str, ...] = tuple(f"v{stratum}_ate" for stratum in NECESSITY_STRATA)
DOUBLE_ROBUST_CONFIGURATIONS = (
    "both_correct",
    "outcome_correct",
    "treatment_correct",
    "both_wrong",
)
NECESSITY_ARMS = ("stratified", "marginal_fluctuation")
JOINT_LABELS = ("strata", "crossfit_strata")


def label_name(label: str) -> str:
    """The package's name of the parameter a property label reads."""
    stratum, key = label.split("_", 2)[:2]
    stem = next(stem for stem, short in KEYS.items() if short == key)
    return law.stratum_name(stem, int(stratum[1:]))


PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in (*CALIBRATION_LABELS, *CROSSFIT_LABELS)
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "simultaneous_coverage": tuple(
        f"{label}__{kind}"
        for label in JOINT_LABELS
        for kind in ("simultaneous_band", "pointwise_joint_control")
    ),
    "double_robustness": tuple(
        f"{label}__{configuration}"
        for label in ATE_LABELS
        for configuration in DOUBLE_ROBUST_CONFIGURATIONS
    ),
    "stratum_targeting_necessity": tuple(
        f"{label}__{arm}" for label in NECESSITY_LABELS for arm in NECESSITY_ARMS
    ),
}


STUDY = StudyRecord(
    name="ordinary point-treatment TMLE with baseline strata",
    slug="canonical-stratified-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle3_stratified",
    document="docs/technical-reference/method-evidence/stratified-point-treatment-tmle.md",
    anchor="stratified-point-treatment-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-stratified-tmle",
    reference="tmle3-stratified",
    modules=(
        "tests/studies/canonical_stratified_tmle.py",
        "tests/studies/stratified_tmle_properties.py",
        "tests/studies/stratified_law.py",
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
    runner_module="tests.studies.canonical_stratified_tmle",
    properties_module="tests.studies.stratified_tmle_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_SD,
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "law": (
        "V in {0, 1, 2} with P(V) = (0.5, 0.3, 0.2); W in {0, 1, 2} given V with rows "
        "(0.5, 0.3, 0.2), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5); g = expit(-0.3 + 0.5 W - 0.4 V); "
        "Q = expit(-0.8 + 0.7 A + 0.6 W + 0.6 V); binary Y"
    ),
    "cross_fit": False,
    "estimands": ["ey", "ate"],
    "strata": ["V"],
    "covariates": list(COVARIATES),
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "outcome_model": "LogisticRegression(C=1e6) on (A, W, V), matched to sl3 Lrnr_glm",
    "treatment_model": "LogisticRegression(C=1e6) on (W, V), matched to sl3 Lrnr_glm",
    "reference_stages": (
        "tmle_stratified(tmle_TSM_all(), base_estimate = FALSE) for the six arm means, then "
        "tmle_stratified(tmle_ATE(1, 0), base_estimate = FALSE) for the three contrasts"
    ),
    "declared_difference": (
        "the tmle3 ATE stage fluctuates one clever-covariate column per stratum; the cleverly "
        "fit with estimands ('ey', 'ate') fluctuates the two arm columns of each stratum"
    ),
    "excluded_comparisons": (
        "ATT and ATC: Param_ATT and Param_ATC read the counterfactual tasks of the training "
        "task rather than of the stratum subset, and they update the A node; PAR: Param_PAR "
        "is not verified as safe on a subset at ed72f8a. Each is measured against exact truth "
        "in the property cells"
    ),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the stratified law from an explicit seed, with every truth."""
    if scenario != SCENARIO:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.sample(n, seed), dict(law.TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def logistic() -> LogisticRegression:
    """The unpenalized main-terms logistic regression both nuisances use."""
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    estimands: Sequence[str] = ("ey", "ate"),
    cross_fit: bool = False,
    simultaneous: bool = False,
    outcome_learner: Any = None,
    treatment_learner: Any = None,
) -> Any:
    """The in-sample stratified fit, or a declared variant of it."""
    return (
        TMLE(
            outcome_learner=logistic() if outcome_learner is None else outcome_learner,
            treatment_learner=logistic() if treatment_learner is None else treatment_learner,
            cross_fit=cross_fit,
            estimands=tuple(estimands),
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
    payload: tuple[int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    replicate, n = payload
    frame, truth = draw_scenario(SCENARIO, n, replicate)
    sample = frame.copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", SCENARIO)
    truth_row = {
        "scenario": SCENARIO,
        "replicate": replicate,
        **{f"truth_{name}": float(truth[name]) for name in ESTIMANDS},
    }
    return sample, truth_row, cleverly_rows(frame, truth, SCENARIO, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw every primary replication, fit the subject, and pack the reference inputs."""
    payloads = [((replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, fitted in outcomes for row in fitted]).loc[
        :, list(REPLICATE_COLUMNS)
    ]
    return samples, truths, rows
