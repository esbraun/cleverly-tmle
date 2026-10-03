"""Registered evidence for DR-TMLE with baseline strata.

A stratified ``DRTMLE`` fit solves every equation of the doubly robust alternation with one
block per stratum and fits every reduced regression inside each stratum.  Each stratum's
estimate is then the DR-TMLE of the law given ``V = s``.  The law is the stratified paper law
of :mod:`tests.studies.stratified_alternating_law`: the complete-data binary law of Benkeser et
al. (2017), Section 5.1, whose stratum shifts both intercepts and the treatment effect.

The primary scenario pairs the cross-fitted stratified fit with pinned R ``drtmle`` 1.1.2 at
``538a3a2``.  ``drtmle`` has no stratified form, so the runner calls it once per stratum subset.  Both sides read the same rows, the same
ten-fold assignment and the subject's own initial nuisance arrays (``Qn``, ``gn``), as
:mod:`tests.studies.canonical_drtmle` pairs them.  Inside a stratum ``drtmle`` then fits its
reduced regressions on the stratum's rows, which is the construction the subject uses.  The nine
stratum names are paired.  The three marginal names are not: a marginal ``drtmle`` call reduces on
``g_n`` alone, while the stratified fit's marginal estimate reduces on ``(g_n, V)``, so the two are
different estimators.  A 20-replication smoke run before the declaration measured that
difference at 1.46 times the paired margin for the marginal ATE, and the stratum names at 0.01 to
0.76.

Declared before any run: the law, n = 2,000, R = 800 primary replications, the shared
``Margins()``, the learners, the seeds below, the property cells of
:mod:`tests.studies.stratified_drtmle_properties`, and ``publication_policy="reporting"``: the DR
instruments already carry finite-sample reds, and every red still gets a diagnosis and an owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_alternating_law as law
from tests.studies.canonical_drtmle import ColumnLogistic, FixedFoldDRTMLE, fixed_folds
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20268821
RESAMPLING_SEED = 20268822
SCENARIO = "stratified_both_correct"
G_BOUNDS = (0.01, 0.99)
N_FOLDS = 10
MAX_OUTER = 100
COVARIATES = ("W1", "W2", "W12", "V")

#: The nine paired names, stratum by stratum.
ESTIMANDS: tuple[str, ...] = tuple(
    f"{stem}[V={s}]" for s in law.STRATA for stem in ("ey[0]", "ey[1]", "ate")
)

#: ``marginal_ate``, then ``v<s>_ate``.  The marginal cells read the same fits as the stratum
#: cells.  The marginal estimate of a stratified fit reduces on ``(g_n, V)``, so it is a
#: different estimator from the unstratified fit, and these cells are its only evidence.
ATE_LABELS: tuple[str, ...] = (
    "marginal_ate",
    *(f"v{stratum}_ate" for stratum in law.STRATA),
)
CALIBRATION_KINDS = ("correctly_specified", "shrunken_se_control", "noise_control")
DOUBLE_ROBUST_CONFIGURATIONS = (
    "both_correct",
    "outcome_correct",
    "treatment_correct",
    "both_wrong",
)

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}" for label in ATE_LABELS for kind in CALIBRATION_KINDS
    ),
    "double_robustness": tuple(
        f"{label}__{configuration}"
        for label in ATE_LABELS
        for configuration in DOUBLE_ROBUST_CONFIGURATIONS
    ),
}


def ate_name(label: str) -> str:
    """The package's name of the ATE a property label reads."""
    return "ate" if label == "marginal_ate" else f"ate[V={int(label[1])}]"


def _stratum_second_moment(stratum: int, truth: float) -> float:
    """``E[D^2 | V = s]`` for the ATE curve centred at ``truth``, by quadrature."""
    from scipy.integrate import quad

    def integrand(w1: float) -> float:
        total = 0.0
        for w2 in (0.0, 1.0):
            g = float(law.paper_propensity(w1, w2, stratum))
            q1 = float(law.paper_outcome(1.0, w1, w2, stratum))
            q0 = float(law.paper_outcome(0.0, w1, w2, stratum))
            total += 0.5 * (q1 * (1 - q1) / g + q0 * (1 - q0) / (1 - g) + (q1 - q0 - truth) ** 2)
        return total / 4.0

    value, _ = quad(integrand, -2.0, 2.0, epsabs=1e-12, epsrel=1e-12, limit=200)
    return float(value)


def stratum_ate_sd(stratum: int) -> float:
    """The efficiency-bound SD of a stratum ATE, ``sqrt(E[D_s^2 | V = s] / P(V = s))``."""
    truth = law.paper_truths()[f"ate[V={stratum}]"]
    return float(np.sqrt(_stratum_second_moment(stratum, truth) / law.PAPER_P_V[stratum]))


def marginal_ate_sd() -> float:
    """The efficiency-bound SD of the marginal ATE, ``sqrt(E[D^2])``.

    ``V`` is a covariate, so the curve is the unstratified ATE curve, centred at the marginal
    ATE inside every stratum.
    """
    truth = law.paper_truths()["ate"]
    return float(
        np.sqrt(
            sum(
                mass * _stratum_second_moment(stratum, truth)
                for stratum, mass in zip(law.STRATA, law.PAPER_P_V, strict=True)
            )
        )
    )


EFFICIENCY_SD = {
    label: marginal_ate_sd() if label == "marginal_ate" else stratum_ate_sd(int(label[1]))
    for label in ATE_LABELS
}

STUDY = StudyRecord(
    name="DR-TMLE with baseline strata",
    slug="canonical-stratified-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_stratified",
    document="docs/technical-reference/method-evidence/stratified-dr-tmle.md",
    anchor="stratified-dr-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-stratified-drtmle",
    reference="drtmle-r-stratified",
    publication_policy="reporting",
    modules=(
        "tests/studies/canonical_stratified_drtmle.py",
        "tests/studies/stratified_drtmle_properties.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/discrete_law.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_stratified_drtmle",
    properties_module="tests.studies.stratified_drtmle_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_SD,
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "drtmle on each stratum subset, shared Qn, gn and folds",
}

CONFIGURATION = {
    "source_law": (
        "Benkeser et al. (2017), Section 5.1, with V in {0, 1, 2} (P = 0.5, 0.3, 0.2): "
        "g = expit(-W1 + 2 W1 W2 + 0.4 - 0.4 V), Q = expit((0.2 + 0.3 V) A - W1 + 2 W1 W2 "
        "- 0.3 + 0.4 V)"
    ),
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "strata": ["V"],
    "guard": ["Q", "g"],
    "reduction": "univariate",
    "reduced_crossfit": "pooled",
    "update_order": "drtmle",
    "max_outer": MAX_OUTER,
    "max_iter": 100,
    "g_bounds": list(G_BOUNDS),
    "folds": (
        "identical unstratified ten-fold assignments supplied to both implementations, drawn "
        "by cleverly.random_partition from the sample's seed alone"
    ),
    "nuisance_models": {
        "correct_outcome": "unpenalized logistic GLM on (A, W1, W2, W12, V, A V)",
        "correct_treatment": "unpenalized logistic GLM on (W1, W2, W12, V)",
        "misspecified": "the same GLM without W12",
        "reduced_Q": "Gaussian GLM",
        "reduced_g": "binomial GLM",
    },
    "excluded_comparisons": (
        "the three marginal names: a marginal drtmle call reduces on g_n over every row, and "
        "the stratified fit's marginal estimate reduces on (g_n, V). The marginal ATE is "
        "measured against its exact truth by the property cells instead"
    ),
    "declared_difference": (
        "the subject's outer loop stops when every stratum's equations meet the stop rule, so "
        "a stratum can take more rounds than R drtmle takes on that stratum's subset alone"
    ),
}


class PaperOutcome(BaseEstimator, ClassifierMixin):
    """The outcome GLM of the stratified paper law: ``(A, W1, W2, W12, V, A V)``.

    The design is ``[A, W1, W2, W12, V]``.  ``correct=False`` drops ``W12``.

    Parameters
    ----------
    correct : bool
        Whether the GLM carries ``W12``.
    """

    def __init__(self, correct: bool = True) -> None:
        self.correct = correct

    def _design(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        columns = [0, 1, 2, 3, 4] if self.correct else [0, 1, 2, 4]
        return np.column_stack([values[:, columns], values[:, 0] * values[:, 4]])

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> PaperOutcome:
        self.model_ = LogisticRegression(
            C=np.inf, max_iter=5000, solver="newton-cholesky", tol=1e-10, random_state=0
        ).fit(self._design(design), target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._design(design)), dtype=float)


def learners(configuration: str) -> tuple[Any, Any]:
    """The outcome and treatment learners of one nuisance configuration."""
    outcome_correct = configuration in {"both_correct", "outcome_correct"}
    treatment_correct = configuration in {"both_correct", "treatment_correct"}
    # The treatment design is [W1, W2, W12, V].
    return PaperOutcome(outcome_correct), ColumnLogistic(None if treatment_correct else (0, 1, 3))


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of the stratified paper law, with its fold assignment and every truth."""
    if scenario != SCENARIO:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    frame = law.paper_sample(n, seed)
    frame["fold"] = fixed_folds(n, seed + 1).assignment
    return frame, law.paper_truths()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, configuration: str = "both_correct") -> Any:
    """The cross-fitted stratified DR-TMLE fit at one nuisance configuration."""
    outcome, treatment = learners(configuration)
    return (
        FixedFoldDRTMLE(
            frame["fold"].to_numpy(dtype=np.int64),
            outcome_learner=outcome,
            treatment_learner=treatment,
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=ColumnLogistic(),
            cross_fit=True,
            n_folds=N_FOLDS,
            stratify_folds="none",
            estimands=("ey", "ate"),
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_outer=MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            guard=("Q", "g"),
        )
        .fit(frame, outcome="Y", treatment="A", covariates=list(COVARIATES), strata=["V"])
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    *,
    result: Any | None = None,
) -> list[dict[str, Any]]:
    """The primary rows of one replication."""
    return primary_rows(
        result=fit_cleverly(frame) if result is None else result,
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
    result = fit_cleverly(frame)
    nuisance = result.repeats[0].nuisance
    sample = frame.copy()
    sample["qn0"] = nuisance.outcome.arms[0.0]
    sample["qn1"] = nuisance.outcome.arms[1.0]
    sample["gn1"] = nuisance.propensity.arm(1.0)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": reference[name]}
        for name in ESTIMANDS
    ]
    return sample, truth_rows, cleverly_rows(frame, reference, scenario, replicate, result=result)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw every replication, fit the subject, and return the rows and arrays R reads."""
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]
