"""Registered evidence for TMLE on a declared known treatment mechanism.

The subject is ``TMLE`` fitted on data that declares its treatment mechanism
(``treatment_probabilities=``), so the fit divides by the design's ``g0`` and fits no
treatment learner.  The law is :mod:`tests.studies.known_mechanism_law`.

**The comparator** is R ``tmle`` 2.1.1 with ``g1W = g0``, on the same realized samples,
through ``tests/canonical/known_mechanism/run_study.R``.  Both sides read the same initial
outcome regression: this package fits it in sample, and the runner receives its arm
predictions as ``Q``.  Both sides bound the mechanism at 0.1 (``gbound = 0.1`` in R, whose
lower-only bound of each arm's total is the two-arm pair ``(0.1, 0.9)`` here).  Every ``g0``
lies in ``[0.25, 0.70]``, so no bound binds on either side.

**The scenarios.**  ``binary_q_correct`` fits the true-terms logistic outcome regression and
reports ``ey0``, ``ey1``, ``ate``, ``rr``, ``or``, ``att`` and ``atc``.  ``binary_q_wrong``
fits the main terms in ``(A, W1, W3)``, which omits the randomization stratum, and reports the
first five.  The two scenarios read the same samples.  R ``tmle`` does not hold a known
mechanism fixed for the ATT and the ATC: it trims the rows, recalibrates ``g1W``, fluctuates
it beside ``Q`` and reports the g-weighted plug-in.  The two constructions are asymptotically
equivalent only at the correct outcome regression, so the ATT and ATC are paired there alone,
as a declared construction difference.

Publication policy is ``reporting``.  The red-cell route was declared before any run: a red
gated cell first gets a defect search (the exact-law tests, a replay of the cell's draws, and
an R comparison on the same draws).  A defect found there is fixed and the study re-run once.
Otherwise the cell stays published red, owned by ``X15-known-mechanism``.  A replication that
raises is never redrawn: a failed replication fails the run.  No budget, margin, law, learner
or seed changes after a verdict is seen.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.interventions import Incremental
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import known_mechanism_law as law
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import initial_estimates, primary_rows

TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261150
RESAMPLING_SEED = 2026115001
IMPLEMENTATION = "cleverly-known-mechanism"
REFERENCE = "tmle-r-known-g"

Q_CORRECT = "binary_q_correct"
Q_WRONG = "binary_q_wrong"
MEANS: tuple[str, ...] = ("ey0", "ey1", "ate", "rr", "or")
CONDITIONAL: tuple[str, ...] = ("att", "atc")
ESTIMANDS: dict[str, tuple[str, ...]] = {Q_CORRECT: (*MEANS, *CONDITIONAL), Q_WRONG: MEANS}
OUTCOME_KIND = {Q_CORRECT: "correct", Q_WRONG: "wrong"}
#: The mechanism bound of both sides.  Every declared value lies in [0.25, 0.70].
G_BOUNDS = (0.1, 0.9)
COVARIATES = ["W1", "W2", "W3"]
#: The declared mechanism, by column, at two and three arms.
PROBABILITIES = {2: {0.0: "p0", 1.0: "p1"}, 3: {0.0: "p0", 1.0: "p1", 2.0: "p2"}}
#: The floor the mean absolute targeting displacement must clear at the wrong outcome
#: regression, on each side: the paired rows then compare two targeting steps rather than two
#: implementations returning their initial fits.
TARGETING_WITNESS_FLOOR = 0.005

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "known_mechanism_accuracy": (
        "ate__known_g__q_correct",
        "ate__known_g__q_wrong",
        "ate__estimated_g_wrong__q_wrong",
        "ate__known_g__q_wrong__cv",
        "att__known_g__q_wrong",
        "atc__known_g__q_wrong",
        "ey_ipsi__known_g__incremental",
        *(
            f"{name}__known_g__multi_arm__q_wrong"
            for name in ("ey0", "ey1", "ey2", "ate_1_vs_0", "ate_2_vs_0")
        ),
    ),
    "bootstrap_coverage": ("ate__known_g__q_wrong__percentile",),
    "interval_calibration": (
        "ate__known_g_q_wrong",
        "ate__shrunken_se_control",
        "ate__noise_control",
    ),
    "variance_direction": ("ate__known_mechanism", "ate__parametric_mechanism"),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
}

STUDY = StudyRecord(
    name="TMLE on a declared known treatment mechanism",
    slug="known-treatment-mechanism",
    artifacts=ROOT / "tests" / "canonical" / "known_mechanism",
    document="docs/technical-reference/method-evidence/known-treatment-mechanism.md",
    anchor="tmle-on-a-declared-known-treatment-mechanism",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={Q_WRONG: Q_CORRECT},
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    modules=(
        "tests/studies/canonical_known_mechanism.py",
        "tests/studies/known_mechanism_properties.py",
        "tests/studies/known_mechanism_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/tmle_mar/Dockerfile",
        "tests/canonical/known_mechanism/run_study.R",
        "tests/canonical/tmle_point_adapter.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_known_mechanism",
    properties_module="tests.studies.known_mechanism_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "tmle(Y, A, W, Q = cbind(qn0, qn1), g1W = g0, gbound = 0.1, family = 'binomial', "
        "cvQinit = FALSE); Q is this package's initial outcome regression"
    ),
    "pairing": (
        "att and atc are paired at the correct outcome regression only: R tmle trims, "
        "recalibrates g1W and fluctuates it for the ATT and the ATC, a construction "
        "asymptotically equivalent to the known-mechanism ATT only when Qbar is correct"
    ),
}

CONFIGURATION = {
    "construction": "ordinary binary TMLE dividing by the declared mechanism",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "treatment_mechanism": "known (declared by column)",
    "outcome_learner": {
        Q_CORRECT: "unpenalized logistic regression on the true terms",
        Q_WRONG: "unpenalized logistic regression on A, W1, W3 (omits W2)",
    },
    "control_population_bias": law.CONTROL_BIAS,
    "targeting_witness_floor": TARGETING_WITNESS_FLOOR,
}


class OutcomeGLM(BaseEstimator, ClassifierMixin):
    """Unpenalized logistic regression on the declared outcome features."""

    def __init__(self, kind: str = "wrong", arms: int = 2) -> None:
        self.kind = kind
        self.arms = arms

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> OutcomeGLM:
        self.model_ = LogisticRegression(penalty=None, max_iter=10_000, tol=1e-10).fit(
            law.outcome_features(design, self.kind, self.arms), target, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        features = law.outcome_features(design, self.kind, self.arms)
        return np.asarray(self.model_.predict_proba(features), dtype=float)


class MechanismGLM(BaseEstimator, ClassifierMixin):
    """Unpenalized logistic regression of the treatment on ``W2``, or intercept-only."""

    def __init__(self, kind: str = "correct") -> None:
        self.kind = kind

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> MechanismGLM:
        self.model_ = LogisticRegression(penalty=None, max_iter=10_000, tol=1e-10).fit(
            law.treatment_features(design, self.kind), target, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        features = law.treatment_features(design, self.kind)
        return np.asarray(self.model_.predict_proba(features), dtype=float)


class DeclaredOnly(BaseEstimator, ClassifierMixin):
    """A treatment learner that must never run: the declared mechanism replaces it."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DeclaredOnly:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")

    def predict_proba(self, design: Any) -> Any:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    outcome: str = "wrong",
    mechanism: str = "known",
    estimands: Sequence[str] | None = ("ate",),
    arms: int = 2,
    cross_fit: bool = False,
    n_bootstrap: int = 0,
    incremental: bool = False,
    random_state: int = 0,
) -> Any:
    """Fit this study's estimator on one sample.

    The primary replications and every property cell call this, so the configuration is
    stated once.  ``mechanism="known"`` declares the mechanism by column; ``"correct"`` and
    ``"intercept"`` estimate it by a logistic regression on ``W2`` or on an intercept.

    Parameters
    ----------
    frame : pandas.DataFrame
        One draw of :func:`tests.studies.known_mechanism_law.sample`.
    outcome : {"correct", "wrong"}
        The outcome regression.
    mechanism : {"known", "correct", "intercept"}
        Whether the mechanism is declared or estimated, and how.
    estimands : sequence of str or None
        The parameters to report, or ``None`` for the axis default.
    arms : int
        2 or 3.
    cross_fit : bool
        Whether to cross-fit the outcome regression over five folds (pooled CV-TMLE).
    n_bootstrap : int
        Full-refit bootstrap replicates, 0 for none.
    incremental : bool
        Whether to fit the incremental intervention at :data:`~tests.studies.known_mechanism_law.DELTA`.
    random_state : int
        The fit's seed, which the bootstrap and the folds read.

    Returns
    -------
    Any
        The single fitted result.
    """
    known = mechanism == "known"
    settings: dict[str, Any] = {
        "outcome_learner": OutcomeGLM(outcome, arms),
        "treatment_learner": DeclaredOnly() if known else MechanismGLM(mechanism),
        "cross_fit": cross_fit,
        "n_folds": 5,
        "estimands": None if estimands is None else tuple(estimands),
        "simultaneous": False,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": random_state,
        "n_bootstrap": n_bootstrap,
    }
    if incremental:
        # An incremental fit refuses a declared g_bounds: g is inside its estimand.
        settings["incremental"] = [Incremental(law.DELTA)]
    else:
        settings["g_bounds"] = G_BOUNDS
    return (
        TMLE(**settings)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_probabilities=PROBABILITIES[arms] if known else None,
        )
        .single()
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of the two-arm law from an explicit seed, with its truth."""
    if scenario not in ESTIMANDS:
        raise KeyError(scenario)
    truth = law.truth(2)
    return law.sample(n, seed), {name: truth[name] for name in ESTIMANDS[scenario]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def conditional_initials(result: Any) -> dict[str, float]:
    """The untargeted ATT and ATC: the arm means of ``Qbar1 - Qbar0`` among the treated and not."""
    treated = np.asarray(result.data.treatment, dtype=float) == 1.0
    blip = np.asarray(result.nuisance.outcome.arms[1.0]) - np.asarray(
        result.nuisance.outcome.arms[0.0]
    )
    return {"att": float(blip[treated].mean()), "atc": float(blip[~treated].mean())}


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> tuple[Any, list[dict[str, Any]]]:
    """Fit one primary replication and transcribe it to the shared schema."""
    estimands = ESTIMANDS[scenario]
    result = fit_cleverly(frame, outcome=OUTCOME_KIND[scenario], estimands=estimands)
    initials = initial_estimates(result, estimands)
    if scenario == Q_CORRECT:
        initials.update(conditional_initials(result))
    rows = primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=estimands,
        initials=initials,
    )
    return result, rows


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result, rows = cleverly_rows(frame, truth, scenario, replicate)
    sample = frame.loc[:, ["W1", "W2", "W3", "A", "Y"]].copy()
    # The comparator reads this package's own initial outcome regression, so the two sides
    # start from one Qbar and differ only in their targeting steps.
    sample["qn0"] = np.asarray(result.nuisance.outcome.arms[0.0], dtype=float)
    sample["qn1"] = np.asarray(result.nuisance.outcome.arms[1.0], dtype=float)
    sample["gn1"] = frame["p1"].to_numpy(dtype=float)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, fit, and retain the samples, initial predictions and mechanism R reads."""
    outcomes = map_parallel(
        _replicate,
        [((scenario, replicate, n),) for scenario in ESTIMANDS for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]


def targeting_displacement(rows: pd.DataFrame) -> dict[str, float]:
    """Mean ``|estimate - initial_estimate|`` of each implementation at the wrong ``Qbar``.

    The nonzero-targeting witness of ``docs/development/method-benchmarking.md``: each value
    must clear :data:`TARGETING_WITNESS_FLOOR`.
    """
    selected = rows.loc[(rows["scenario"] == Q_WRONG) & (rows["estimand"] == "ate")]
    return {
        str(name): float(np.mean(np.abs(group["estimate"] - group["initial_estimate"])))
        for name, group in selected.groupby("implementation")
        if not math.isnan(float(group["initial_estimate"].iloc[0]))
    }
