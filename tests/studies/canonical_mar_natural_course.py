"""Registered evidence study for the natural-course mean under MAR.

The comparator is R ``tmle`` 2.1.1 through its population-mean path, with the same supplied
outcome and response predictions.  The continuous law plants the two ``q_bounds`` on two
non-responding rows so that R takes the declared outcome scale, and ``scale-probe.csv``
records the exact-equality probe of that workaround on every continuous replication.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleOutcomeContinuous
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.missing_outcome_study_helpers import (
    FailTreatment,
    NaturalCourseLaw,
    efficiency_sd,
    sample_discrete,
)
from tests.studies.point_study_helpers import natural_course_initial_estimate, primary_rows

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20261007
SCENARIOS = ("binary_mar_natural_course", "continuous_mar_natural_course")
ESTIMANDS = ("ey_obs",)
Q_BOUNDS = (0.0, 1.0)
NUISANCE_BOUND = 0.01
BETA_CONCENTRATION = 24.0
TRUTH = float(mar.functional(mar.PROBS, "ey_obs"))
EFFICIENCY_SD = efficiency_sd(mar.PROBS, "ey_obs")
CONTINUOUS_SCENARIO = "continuous_mar_natural_course"
TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)


STUDY = StudyRecord(
    name="ordinary missing-outcome natural-course TMLE",
    slug="mar-natural-course-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_mar_natural_course",
    document=(
        "docs/technical-reference/method-evidence/ordinary-missing-outcome-natural-course-tmle.md"
    ),
    anchor="ordinary-missing-outcome-natural-course-tmle",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261008,
    margins=Margins(),
    implementation="cleverly-mar-natural-course-tmle",
    reference="tmle-r-population-mean",
    modules=(
        "tests/studies/canonical_mar_natural_course.py",
        "tests/studies/mar_natural_course_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/discrete_law_mar.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mar_natural_course",
    properties_module="tests.studies.mar_natural_course_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "response_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ey_obs__correctly_specified",
            "ey_obs__shrunken_se_control",
            "ey_obs__noise_control",
        ),
        "targeting_necessity": ("ey_obs__targeted", "ey_obs__untargeted"),
        "missingness_necessity": ("ey_obs__declared", "ey_obs__complete_case_control"),
    },
    efficiency_bounds={"ey_obs": EFFICIENCY_SD},
    extra_artifacts=("scale-probe.csv",),
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_inference": (
        "native R tmle centered sample variance; bounded binary-outcome interval"
    ),
}

CONFIGURATION = {
    "construction": "ordinary MAR natural-course TMLE",
    "source": "Diaz, Carone, and van der Laan (2016), Section 2, equations 1-5",
    "cross_fit": False,
    "fluctuation": "logistic",
    "targeting": "iterative",
    "target_weights": False,
    "simultaneous_intervals": False,
    "q_bounds": list(Q_BOUNDS),
    "missingness_bound": NUISANCE_BOUND,
    "treatment_mechanism": "deliberately unavailable and asserted unused",
    "nuisance_predictions": (
        "exact response probabilities and binary outcome means; correctly specified "
        "affine outcome learner for the continuous law"
    ),
    "continuous_outcome": f"beta with concentration {BETA_CONCENTRATION:g}",
    "reference_arguments": (
        f"A=rep(1, n), W=data.frame(A_original, W), Q=cbind(qn, qn), g1W=rep(1, n), "
        f"pDelta1=cbind(pin, pin), fluctuation=logistic, Qbounds=c(0, 1), "
        f"gbound={NUISANCE_BOUND}, alpha=0.9995, cvQinit=FALSE, prescreenW.g=FALSE, "
        f"target.gwt=FALSE, evalATT=FALSE, B=1; binomial for the binary law, gaussian for "
        f"the continuous law"
    ),
    "reference_scale_workaround": (
        "R tmle takes the continuous outcome range from every non-NA Y, so each continuous "
        "fit sets Y to the two q_bounds on two rows that the fluctuation and the curve "
        "exclude; scale-probe.csv records the exact-equality probe of that workaround on "
        "every continuous replication"
    ),
    "comparator_search": (
        "Used R tmle 2.1.1 through its population-mean path, with a constant synthetic "
        "treatment and the same supplied predictions as this fit: the exact binary outcome "
        "means, the affine continuous outcome learner's predictions, and the exact response "
        "probabilities. The comparison conditions on those supplied predictions. Its native "
        "var(IC)/n uses a centered n-1 sample variance, as this non-cross-fitted estimator's "
        "centered rule does. The stacked natural-course study's agreement is not transferred "
        "to this estimator."
    ),
}


def _sample_continuous(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.choice(len(mar.P_W), size=n, p=mar.P_W)
    a = rng.binomial(1, mar.G[w])
    delta = rng.binomial(1, mar.PI[w, a])
    mean = mar.Q[w, a]
    y = rng.beta(mean * BETA_CONCENTRATION, (1.0 - mean) * BETA_CONCENTRATION)
    return pd.DataFrame(
        {
            "W": w.astype(float),
            "A": a.astype(float),
            "Y": np.where(delta == 1, y, np.nan),
            "Delta": delta.astype(float),
        }
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario == "binary_mar_natural_course":
        frame = sample_discrete(mar.PROBS, n, seed)
    elif scenario == "continuous_mar_natural_course":
        frame = _sample_continuous(n, seed)
    else:
        raise KeyError(scenario)
    return frame, {"ey_obs": TRUTH}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, scenario: str, *, law: NaturalCourseLaw | None = None) -> Any:
    declared = NaturalCourseLaw() if law is None else law
    outcome = (
        OracleOutcome(declared)
        if scenario == "binary_mar_natural_course"
        else OracleOutcomeContinuous(declared)
    )
    return (
        TMLE(
            estimands=ESTIMANDS,
            outcome_learner=outcome,
            treatment_learner=FailTreatment(),
            missingness_learner=OracleMissingness(declared),
            cross_fit=False,
            fluctuation="logistic",
            targeting="iterative",
            target_weights=False,
            simultaneous=False,
            **({"q_bounds": Q_BOUNDS} if scenario == "continuous_mar_natural_course" else {}),
            nuisance_bound=NUISANCE_BOUND,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta")
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    result: Any | None = None,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario) if result is None else result
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials={"ey_obs": natural_course_initial_estimate(result)},
    )


def reference_sample(
    frame: pd.DataFrame,
    result: Any,
    *,
    scenario: str,
    replicate: int,
) -> pd.DataFrame:
    """Serialize the exact supplied predictions that the R population-mean path targets.

    ``qn`` is the unscaled initial outcome prediction at each realized arm, and ``pin`` is the
    response prediction at each realized arm.  The payload refuses a fit whose response bound
    binds, because R would then target a different nuisance.
    """
    missingness = result.nuisance.missingness_at_realised_arm(result.data.treatment)
    if missingness is None:  # pragma: no cover - the registered fit declares Delta
        raise AssertionError("the reference payload needs fitted response predictions")
    response = np.asarray(missingness, dtype=float)
    if np.any(response <= NUISANCE_BOUND):
        raise AssertionError(f"{scenario}/{replicate}: the response bound binds")
    outcome = result.nuisance.scaler.unscale_levels(result.nuisance.outcome.observed)
    return pd.DataFrame(
        {
            "scenario": scenario,
            "replicate": replicate,
            "W": frame["W"].to_numpy(dtype=float),
            "A": frame["A"].to_numpy(dtype=float),
            "Y": frame["Y"].to_numpy(dtype=float),
            "Delta": frame["Delta"].to_numpy(dtype=float),
            "qn": np.asarray(outcome, dtype=float),
            "pin": response,
        }
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
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
        [((scenario, replicate, n),) for scenario in SCENARIOS for replicate in range(replicates)],
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
PROBE_RUNNER = "tmle_mar_natural_course/probe_scale_workaround.R"
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
    """Run the exact-equality probe of the planted-row scale workaround on every continuous fit."""
    path = output / PROBE_ARTIFACT
    reference.run(here, samples, truths_path, path, cores=cores, runner=PROBE_RUNNER)
    probe = pd.read_csv(path).sort_values("replicate", ignore_index=True)
    reference_rows = pd.read_csv(reference_results, usecols=["scenario", "replicate"])
    expected = (
        reference_rows.loc[reference_rows["scenario"] == CONTINUOUS_SCENARIO, "replicate"]
        .sort_values()
        .reset_index(drop=True)
    )
    if not probe["replicate"].equals(expected):
        raise RuntimeError("the scale-workaround probe does not match the reference replications")
    return {PROBE_ARTIFACT: probe}


def scientific_failures(extra_frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Refuse publication when the continuous comparison's scale workaround is not exact.

    Coverage requires exactly one probe row for each continuous replication ``0, ..., k - 1``.
    The artifact hook also compares those rows with the reference's actual replications.
    """
    probe = extra_frames[PROBE_ARTIFACT]
    if probe.empty:
        return {"scale-workaround probe coverage": pd.DataFrame([{"error": "no probe rows"}])}
    replicates = sorted(int(value) for value in probe["replicate"])
    if set(probe["scenario"]) != {CONTINUOUS_SCENARIO} or replicates != list(range(len(probe))):
        return {"scale-workaround probe coverage": probe}
    return {"scale-workaround probe": probe.loc[~probe["passed"].fillna(False).eq(True)]}
