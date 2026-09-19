"""Registered evidence for stacked CV-TMLE of arm-indexed means and contrasts under MAR.

``docs/roadmap.md`` RM9 ("Registered study") is the contract this module implements.  Four
finite-covariate laws, declared in :mod:`tests.studies.mar_arm_indexed_laws`, each draw 800
samples of 2,000 rows.  Every sample is fitted once with ten unstratified folds, and the fit
reports every estimand the law declares.

The comparator is R ``tmle`` 2.1.1 with the same stitched out-of-fold predictions.  The
two-arm laws use its native two-arm path.  The three-arm laws run its population-mean path once
for each arm, and the runner builds the joint covariance and each reference contrast from the
returned per-arm influence curves.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly import CrossFitting, Inference, ModelSpec, Runtime, Targeting, TMLEMethod
from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20261301
N_FOLDS = 10
NUISANCE_BOUND = 0.01
G_BOUNDS = (0.01, 0.99)
#: The scalar R ``tmle`` bound on each supplied product ``g_a pi_a``.  R turns it into
#: ``c(R_GBOUND, 1)``, and the runner asserts that no supplied product falls below it.
R_GBOUND = 0.001
TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
#: The widest arm count any law declares, which fixes the payload's per-arm columns.
MAX_ARMS = 3

SCENARIOS: dict[str, tuple[str, ...]] = {
    law.scenario: laws.ESTIMANDS[key] for key, law in laws.LAWS.items()
}
#: The exact efficient standard deviation of every reported estimand, keyed by its
#: property-cell label because two laws report estimands of the same name.
EFFICIENCY_SD: dict[str, float] = {
    laws.label(law, name): laws.efficiency_sd(law, name)
    for key, law in laws.LAWS.items()
    for name in laws.ESTIMANDS[key]
}
LABELS: tuple[str, ...] = tuple(EFFICIENCY_SD)
#: The contrast each law's robustness and overfitting cells follow.
ATE_LABELS: tuple[str, ...] = tuple(label for label in LABELS if "_ate" in label)
OVERFIT_LABELS: tuple[str, ...] = ("l1_ate", "l2_ate", "l3_ate_low", "l4_ate_low")
ROBUSTNESS_CONFIGURATIONS: tuple[str, ...] = (
    "only_outcome_wrong",
    "only_treatment_wrong",
    "only_response_wrong",
    "outcome_and_response_wrong",
)


STUDY = StudyRecord(
    name="stacked arm-indexed missing-outcome CV-TMLE",
    slug="stacked-mar-arm-indexed-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_mar_arm_indexed_cvtmle",
    document=(
        "docs/technical-reference/method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md"
    ),
    anchor="stacked-arm-indexed-missing-outcome-cv-tmle",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    nuisance_count=3,
    resampling_seed=20261302,
    margins=Margins(),
    implementation="cleverly-stacked-mar-arm-indexed-cvtmle",
    reference="tmle-r-stitched-arm-indexed",
    modules=(
        "tests/studies/canonical_mar_arm_indexed_cvtmle.py",
        "tests/studies/mar_arm_indexed_cvtmle_properties.py",
        "tests/studies/mar_arm_indexed_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law_mar.py",
        "tests/discrete_law_multi.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mar_arm_indexed_cvtmle",
    properties_module="tests.studies.mar_arm_indexed_cvtmle_properties",
    property_cells={
        "interval_calibration": tuple(
            f"{label}__{kind}"
            for label in LABELS
            for kind in ("learned_nuisances", "shrunken_se_control")
        ),
        "simultaneous_coverage": tuple(
            f"{key}__{kind}"
            for key in laws.LAWS
            for kind in ("simultaneous_band", "pointwise_joint_control")
        ),
        "mar_robustness": tuple(
            f"{label}__{configuration}"
            for label in ATE_LABELS
            for configuration in ROBUSTNESS_CONFIGURATIONS
        ),
        "crossfit_overfitting": tuple(
            f"{label}__{kind}"
            for label in OVERFIT_LABELS
            for kind in ("stacked_arm_indexed_cvtmle", "in_sample_control")
        ),
    },
    # The flexible learned-nuisance cells do not claim efficiency-bound attainment.  The
    # bounds are published so a reader can set each against the measured standard error.
    efficiency_bounds=EFFICIENCY_SD,
    calibration_efficiency_ratio=False,
    extra_artifacts=("scale-probe.csv",),
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_inference": (
        "native R tmle two-arm results for L1 and L2; for L3 and L4, Wald intervals from the "
        "centered n-1 covariance of the per-arm population-mean influence curves"
    ),
}

CONFIGURATION = {
    "construction": "one-repeat stacked CV-TMLE of arm-indexed means and contrasts under MAR",
    "source": (
        "Díaz and van der Laan (2017); Gruber and van der Laan (2012), Section 2.3; "
        "Zheng and van der Laan (2011), Theorem 2; Levy (2018), Section 3.1"
    ),
    "laws": {
        law.scenario: {
            "arms": law.arms,
            "outcome": "bounded continuous" if law.continuous else "binary",
            "q_bounds": None if law.q_bounds is None else list(law.q_bounds),
            "requested_estimands": list(laws.REQUESTS[key]),
        }
        for key, law in laws.LAWS.items()
    },
    "continuous_outcome": (
        f"Y = {laws.LOWER} + {laws.UPPER - laws.LOWER} * Beta({laws.PHI} mu, {laws.PHI} (1 - mu))"
    ),
    "reference_arm": laws.REFERENCE,
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "repeats": 1,
    "stratify_folds": "none",
    "targeting_scheme": "pooled",
    "fold_evaluation": False,
    "split_plan": None,
    "fluctuation": "logistic",
    "targeting": "iterative",
    "target_weights": False,
    "max_iter": 100,
    "tol": 1e-10,
    "g_bounds": list(G_BOUNDS),
    "missingness_bound": NUISANCE_BOUND,
    "simultaneous_intervals": False,
    "band_cells": (
        "the calibration fits declare Inference(simultaneous=True) with its default multiplier "
        "draws, which leaves every pointwise result unchanged and attaches the band"
    ),
    "primary_nuisance_predictions": (
        "separate depth-five trees with minimum leaf size 25 for the outcome regression "
        "(classification for a binary outcome, regression for a continuous one), the "
        "treatment mechanism, and the response mechanism"
    ),
    "reference_arguments": (
        f"cvQinit=FALSE, prescreenW.g=FALSE, target.gwt=FALSE, evalATT=FALSE, B=1, "
        f"alpha=0.9995, gbound={R_GBOUND}; binomial for L1 and L3, gaussian with Qbounds "
        f"equal to q_bounds for L2 and L4"
    ),
    "reference_scale_workaround": (
        "R tmle takes the continuous outcome range from every non-NA Y, so each continuous "
        "fit sets Y to the two q_bounds on two rows that the fluctuation and the curves "
        "exclude; scale-probe.csv records the exact-equality probe of that workaround"
    ),
    "comparator_search": (
        "Used R tmle 2.1.1 with the same stitched out-of-fold outcome, treatment, and "
        "response predictions and one pooled fluctuation: its native two-arm path for L1 "
        "and L2, and its population-mean path once per arm for L3 and L4, with the joint "
        "covariance built from the per-arm influence curves. Its native var(IC)/n uses a "
        "centered n-1 sample variance, as this estimator's centered rule does. Rejected "
        "tmle3 at ed72f8a because its generic treatment-specific outcome fit can use the "
        "Delta = 0 pseudo-outcomes when it predicts under Delta = 1, while this estimator "
        "fits that regression on respondents only; rejected zEpid 0.9.1 because its "
        "cross-fit TMLE targets inside each fold rather than with one pooled fluctuation; "
        "rejected Newey and Robins (2018) because their construction fits the outcome and "
        "inverse response regressions on distinct subsamples, which is a different estimator."
    ),
}


# ------------------------------------------------------------------------------ learners


def flexible_outcome_learner(law: laws.Law) -> Any:
    """The predeclared data-adaptive outcome learner."""
    if law.continuous:
        return DecisionTreeRegressor(max_depth=5, min_samples_leaf=25, random_state=0)
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=0)


def flexible_treatment_learner() -> DecisionTreeClassifier:
    """The separately fitted, predeclared data-adaptive treatment learner."""
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=1)


def flexible_response_learner() -> DecisionTreeClassifier:
    """The separately fitted, predeclared data-adaptive response learner."""
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=2)


# ------------------------------------------------------------------------------ sampling


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    law = laws.BY_SCENARIO[scenario]
    return laws.sample(law, n, seed), laws.truths(law)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ------------------------------------------------------------------------------- fitting


def method(
    law: laws.Law,
    *,
    outcome_learner: Any,
    treatment_learner: Any,
    missingness_learner: Any,
    cross_fit: bool = True,
    simultaneous: bool = False,
) -> TMLEMethod:
    """Build the exact public RM9 method declaration used by every study arm."""
    return TMLEMethod(
        models=ModelSpec(
            outcome_learner=outcome_learner,
            treatment_learner=treatment_learner,
            missingness_learner=missingness_learner,
        ),
        cross_fitting=CrossFitting(
            enabled=cross_fit,
            n_folds=N_FOLDS,
            repeats=1,
            # The in-sample control is outside the cross-fit contract, where
            # ``stratify_by="none"`` stays reserved.  Its folds are never used.
            stratify_by="none" if cross_fit else "treatment",
            targeting_scheme="pooled",
            fold_evaluation=False,
            split_plan=None,
        ),
        targeting=Targeting(
            fluctuation="logistic",
            algorithm="iterative",
            g_bounds=G_BOUNDS,
            q_bounds=law.q_bounds,
            nuisance_bound=NUISANCE_BOUND,
            target_weights=False,
            max_iter=100,
            tol=1e-10,
        ),
        inference=Inference(simultaneous=simultaneous),
        runtime=Runtime(random_state=0, n_jobs=1),
    )


def fit_cleverly(
    frame: pd.DataFrame,
    law: laws.Law,
    *,
    outcome_learner: Any | None = None,
    treatment_learner: Any | None = None,
    missingness_learner: Any | None = None,
    cross_fit: bool = True,
    simultaneous: bool = False,
    adjustment: tuple[str, ...] = ("W",),
) -> Any:
    """Fit every estimand a law declares, with one fold draw and one pooled fluctuation."""
    configured = method(
        law,
        outcome_learner=(
            flexible_outcome_learner(law) if outcome_learner is None else outcome_learner
        ),
        treatment_learner=(
            flexible_treatment_learner() if treatment_learner is None else treatment_learner
        ),
        missingness_learner=(
            flexible_response_learner() if missingness_learner is None else missingness_learner
        ),
        cross_fit=cross_fit,
        simultaneous=simultaneous,
    )
    estimator = TMLE(
        **configured.estimator_kwargs(),
        estimands=laws.REQUESTS[law.key],
        reference=laws.REFERENCE if law.arms == 3 else None,
    )
    return estimator.fit(
        frame, outcome="Y", treatment="A", covariates=list(adjustment), delta="Delta"
    ).single()


def initial_estimates(result: Any, law: laws.Law) -> dict[str, float]:
    """Each reported estimand's untargeted plug-in, on its natural scale."""
    nuisance = result.nuisance
    psi = np.empty(law.arms)
    for code in result.data.arm_codes:
        values = nuisance.scaler.unscale_levels(nuisance.outcome.arms[code])
        psi[law.column(int(code))] = float(np.mean(np.asarray(values, dtype=float)))
    return laws.values(law, psi)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    result: Any | None = None,
) -> list[dict[str, Any]]:
    law = laws.BY_SCENARIO[scenario]
    result = fit_cleverly(frame, law) if result is None else result
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=laws.ESTIMANDS[law.key],
        initials=initial_estimates(result, law),
    )


def reference_sample(
    frame: pd.DataFrame,
    result: Any,
    *,
    scenario: str,
    replicate: int,
) -> pd.DataFrame:
    """Serialize the stitched predictions and folds that the R comparator targets.

    Columns ``q{c}``, ``g{c}`` and ``pi{c}`` hold the unscaled outcome regression, the
    treatment mechanism, and the response mechanism at arm code ``c``.  A two-arm law
    leaves the third set empty.  The payload refuses a fit whose treatment or response
    bound binds, because R would then target different nuisances.
    """
    law = laws.BY_SCENARIO[scenario]
    nuisance = result.nuisance
    truncation = nuisance.propensity.truncate(G_BOUNDS)
    if np.any(truncation.clipped):
        raise AssertionError(f"{scenario}/{replicate}: the treatment bound binds")
    missingness = np.asarray(nuisance.missingness, dtype=float)
    if np.any(missingness < NUISANCE_BOUND):
        raise AssertionError(f"{scenario}/{replicate}: the response bound binds")
    codes = result.data.arm_codes
    sample = pd.DataFrame(
        {
            "scenario": scenario,
            "replicate": replicate,
            "W": frame["W"].to_numpy(dtype=float),
            "A": np.asarray(result.data.treatment, dtype=int),
            "Y": frame["Y"].to_numpy(dtype=float),
            "Delta": frame["Delta"].to_numpy(dtype=float),
            "fold": np.asarray(nuisance.folds.assignment, dtype=int),
        }
    )
    for code in range(MAX_ARMS):
        if code < law.arms:
            arm = codes[code]
            sample[f"q{code}"] = np.asarray(
                nuisance.scaler.unscale_levels(nuisance.outcome.arms[arm]), dtype=float
            )
            sample[f"g{code}"] = nuisance.propensity.arm(arm)
            sample[f"pi{code}"] = missingness[:, code]
        else:
            sample[f"q{code}"] = np.nan
            sample[f"g{code}"] = np.nan
            sample[f"pi{code}"] = np.nan
    return sample


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, laws.BY_SCENARIO[scenario])
    sample = reference_sample(frame, result, scenario=scenario, replicate=replicate)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, cleverly_rows(frame, truth, scenario, replicate, result=result)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [
            ((scenario, replicate, n),)
            for scenario in STUDY.scenarios
            for replicate in range(replicates)
        ],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]


# ------------------------------------------------------------------ scale-workaround probe

#: The probe runner.  It is not named ``run_*.R``, because
#: ``tests/unit/test_canonical_runner_parity.py`` holds every such file to the published-row
#: contract, and the probe publishes a check table instead.
PROBE_RUNNER = "tmle_mar_arm_indexed_cvtmle/probe_scale_workaround.R"
PROBE_ARTIFACT = "scale-probe.csv"


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
    """Run the exact-equality probe of the planted-row scale workaround on replication 0."""
    del reference_results  # the probe writes its own table rather than reading the rows
    path = output / PROBE_ARTIFACT
    reference.run(here, samples, truths_path, path, cores=cores, runner=PROBE_RUNNER)
    return {PROBE_ARTIFACT: pd.read_csv(path)}


def scientific_failures(extra_frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Refuse publication when the continuous comparison's scale workaround is not exact."""
    probe = extra_frames[PROBE_ARTIFACT]
    expected = {law.scenario for law in laws.LAWS.values() if law.continuous}
    if set(probe["scenario"]) != expected:
        return {"scale-workaround probe coverage": probe}
    return {"scale-workaround probe": probe.loc[~probe["passed"].astype(bool)]}
