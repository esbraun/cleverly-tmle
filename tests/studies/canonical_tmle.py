"""The canonical point-treatment TMLE evidence study.

This module is the study's *declaration*: the two laws it samples from, the parameter
oracles, the non-cross-fitted estimator configuration matched to ordinary R ``tmle3``, and
the margins its verdicts are bounded by.  Everything that turns those into summaries,
verdicts, negative controls and a manifest lives in :mod:`tests.studies.evidence` and is
shared with every method that gets an evidence row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._typing import EstimandName
from cleverly.datasets import DGP, binary_outcome_dgp
from cleverly.estimators import TMLE
from cleverly.utils.bounds import expit
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

#: The pinned reference.  Recorded in the manifest and reproduced by the fixture container.
TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 1000
SEED = 20240819

#: Typed as estimand names rather than plain strings, so the study's own calls exercise the
#: static contract ``TMLE`` declares instead of quietly widening past it.
COMMON_ESTIMANDS: tuple[EstimandName, ...] = ("ey1", "ey0", "ate", "att", "atc", "ey_obs", "par")
BINARY_ESTIMANDS: tuple[EstimandName, ...] = (*COMMON_ESTIMANDS, "paf", "rr", "or")

#: The same map the record carries, keeping the estimand type the record widens away.
SCENARIO_ESTIMANDS: Mapping[str, tuple[EstimandName, ...]] = {
    "binary": BINARY_ESTIMANDS,
    "continuous": COMMON_ESTIMANDS,
}

G_BOUNDS = (0.01, 0.99)

STUDY = StudyRecord(
    name="ordinary point-treatment TMLE",
    slug="canonical-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle3",
    document="docs/technical-reference/method-evidence.md",
    anchor="canonical-point-treatment-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="tmle3",
    # ``cleverly`` reports PAF on the identity scale from the PAF influence curve; ``tmle3``
    # reports a negative-log-complement (log-risk-ratio) quantity mapped through
    # ``1 - exp(-x)``.  The two have the same first-order delta-method limit but are not the
    # same reported scale, so a raw standard-error comparison would compare two things.
    incomparable_se=frozenset({"paf"}),
    modules=(
        "tests/studies/canonical_tmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_tmle",
    properties_module="tests.studies.canonical_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "treatment_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified",),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
    },
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
}


def continuous_dgp() -> DGP:
    """A bounded continuous outcome with effect modification and comfortable overlap."""

    def propensity(w: np.ndarray) -> np.ndarray:
        return expit(0.6 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        return expit(-0.5 + 0.7 * w[:, 0] - 0.4 * w[:, 1] + a * (0.8 + 0.3 * w[:, 0]))

    return DGP(
        name="canonical_tmle_continuous",
        n_latent=2,
        covariate_names=("W1", "W2"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        family="continuous",
        noise_scale=0.08,
    )


def sample_continuous(dgp: DGP, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw a bounded continuous outcome without clipping the conditional mean.

    The Beta draw has ``outcome_mean`` as its conditional mean exactly, so the oracle below
    is the truth of the law that was sampled rather than of a nearby one.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, dgp.n_latent))
    g = np.asarray(dgp.propensity(w), dtype=float)
    a = rng.binomial(1, g).astype(float)
    q1 = np.asarray(dgp.outcome_mean(w, 1.0, None), dtype=float)
    q0 = np.asarray(dgp.outcome_mean(w, 0.0, None), dtype=float)
    mean = np.where(a == 1.0, q1, q0)
    concentration = 24.0
    y = rng.beta(mean * concentration, (1.0 - mean) * concentration)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    return frame, truth_for(dgp)


def truth_for(dgp: DGP) -> dict[str, float]:
    """Every shared estimand, including natural-course quantities absent from ``DGP.truth``.

    Integrated on the same quadrature rule the law's own truths use, so the oracle and the
    sampler cannot disagree about which law this is.
    """
    w = dgp.quadrature()
    g = np.asarray(dgp.propensity(w), dtype=float)
    q1 = np.asarray(dgp.outcome_mean(w, 1.0, None), dtype=float)
    q0 = np.asarray(dgp.outcome_mean(w, 0.0, None), dtype=float)
    contrast = q1 - q0
    ey1 = float(np.mean(q1))
    ey0 = float(np.mean(q0))
    ey_obs = float(np.mean(g * q1 + (1.0 - g) * q0))
    out = {
        "ey1": ey1,
        "ey0": ey0,
        "ate": float(np.mean(contrast)),
        "att": float(np.sum(g * contrast) / np.sum(g)),
        "atc": float(np.sum((1.0 - g) * contrast) / np.sum(1.0 - g)),
        "ey_obs": ey_obs,
        "par": ey_obs - ey0,
    }
    if dgp.family == "binomial":
        out.update(
            {
                "paf": 1.0 - ey0 / ey_obs,
                "rr": ey1 / ey0,
                "or": (ey1 / (1.0 - ey1)) / (ey0 / (1.0 - ey0)),
            }
        )
    return out


def scenario_dgp(scenario: str) -> DGP:
    if scenario == "continuous":
        return continuous_dgp()
    if scenario == "binary":
        return binary_outcome_dgp()
    raise KeyError(scenario)


def draw_for(
    record: StudyRecord, scenario: str, n: int, replicate: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario`` from ``record``'s own seed stream.

    The laws are shared -- the CV studies estimate the same parameters of the same two
    processes -- but the *samples* are not, and the record is what decides which.  Taking the
    record as an argument rather than closing over one is the point: a study that imported a
    ready-made ``draw_scenario`` would silently inherit the seed of whichever module defined
    it, publish its own in ``manifest.json``, and be reproducible from neither.
    """
    return draw_from_seed(scenario, n, replicate_seed(record, scenario, replicate))


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``scenario`` from an explicit seed.

    Part of the runner contract rather than a private helper.  The published-seed audit in
    ``tests/unit/test_method_evidence.py`` redraws a study's first replication and requires
    the committed sample back, and it has to do that *through the study's own module*: a
    study whose scenarios are not the two named here would otherwise be audited against
    this module's laws, which is a check on nothing.
    """
    dgp = scenario_dgp(scenario)
    if scenario == "continuous":
        return sample_continuous(dgp, n, seed)
    frame, _ = dgp.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(dgp)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``: a fixed sample, whatever the study's size."""
    return draw_for(STUDY, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    """The explicitly non-cross-fitted configuration matched to ordinary R ``tmle3``."""
    binary = scenario == "binary"
    outcome = (
        LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs") if binary else LinearRegression()
    )
    treatment = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    estimands = SCENARIO_ESTIMANDS[scenario]
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(
            outcome_learner=outcome,
            treatment_learner=treatment,
            cross_fit=False,
            estimands=estimands,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """One replication's rows in the shared per-replication schema."""
    result = fit_cleverly(frame, scenario)
    rows: list[dict[str, Any]] = []
    for name, estimate in result.estimates.items():
        reference = float(truth[name])
        low, high = estimate.ci
        inference_estimate = (
            float(estimate.log_psi)
            if estimate.scale == "ratio" and estimate.log_psi is not None
            else float(estimate.psi)
        )
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": inference_estimate,
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if estimate.scale == "ratio" else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Draw one replication and fit it, as a picklable unit of work."""
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    payload_frame = frame.copy()
    payload_frame.insert(0, "replicate", replicate)
    payload_frame.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return payload_frame, truth_row, cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every replication's sample, its truth, and every ``cleverly`` row.

    The samples are what the R container is handed, so both implementations see the same
    realized datasets and the comparison is paired by construction.  Truth travels beside
    them, one row per replication rather than repeated on every observation: ten constant
    columns over millions of rows was most of the reference container's memory, and memory is
    what decides how many of its workers fit.
    """
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([frame for frame, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, rows.loc[:, list(REPLICATE_COLUMNS)]
