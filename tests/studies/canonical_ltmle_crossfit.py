"""Canonical cross-fitted end-of-study longitudinal TMLE evidence study.

The registered row compares with pinned ``lmtp`` on identical panels, exact mechanisms, and
rowwise folds. Statistical properties are checked independently against the finite-support law.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import RULE_LABEL, make_longitudinal, rule_arm_at_node_two
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism, QuasiBinomialGLM
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

LMTP_VERSION = "1.5.4"
LMTP_SOURCE_COMMIT = "f04a2b47f46debc515ce4ae778e05ebfde922c44"
LMTP_TARBALL_SHA256 = "fd49d9f291d4ddabb78c36d152b25aaa234a7204b645b9921f998c152e3d2ba5"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20260824
SCENARIO = "censored_end_of_study"
G_BOUNDS = (1e-8, 1.0)

REGIMENS: dict[str, Any] = {
    "never": 0,
    "always": 1,
    RULE_LABEL: (1, lambda history: rule_arm_at_node_two(history["L2"])),
}
REFERENCE = "never"
MEAN_NAMES = tuple(f"ey_regimen[{label}]" for label in REGIMENS)
CONTRAST_NAMES = (
    "ate_regimen[always vs never]",
    f"ate_regimen[{RULE_LABEL} vs never]",
)
ESTIMANDS = (*MEAN_NAMES, *CONTRAST_NAMES)

STUDY = StudyRecord(
    name="cross-fitted end-of-study longitudinal TMLE",
    slug="canonical-ltmle-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_ltmle",
    document="docs/technical-reference/method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md",
    anchor="cross-fitted-end-of-study-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/ltmle_crossfit_properties.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit/audit.py",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_ltmle/run_study.R",
    ),
    runner_module="tests.studies.canonical_ltmle_crossfit",
    properties_module="tests.studies.ltmle_crossfit_properties",
    property_cells={
        "double_robustness": tuple(
            f"{estimand}__{configuration}"
            for estimand in ("static", "dynamic")
            for configuration in (
                "both_correct",
                "outcome_correct",
                "mechanism_correct",
                "both_wrong",
            )
        ),
        "root_n_and_efficiency": tuple(
            f"{estimand}__n_{size}"
            for estimand in ("static", "dynamic")
            for size in (1000, 2000, 8000)
        ),
        "root_n_rate": tuple(
            f"{estimand}__{statistic}"
            for estimand in ("static", "dynamic")
            for statistic in ("empirical_sd", "reported_se")
        ),
        "interval_calibration": tuple(
            f"{estimand}__{cell}"
            for estimand in ("static", "dynamic")
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
        "type_i_error": ("static__sharp_null",),
        "power": ("static__alternative",),
        "targeting_necessity": tuple(
            f"{estimand}__{arm}"
            for estimand in ("static", "dynamic")
            for arm in ("targeted", "untargeted")
        ),
        "crossfit_overfitting": ("cross_fitted_ltmle", "in_sample_control"),
    },
)

#: Provenance of the comparator, for the manifest's ``generated_with.reference`` block.
#:
#: ``write_manifest`` records this block for the declared ``lmtp`` comparator.
REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "fold_specific_cross_fit",
    "outcome_kind": "end_of_study",
    "horizon_mode": "terminal_only",
    "r_survival_outcome": False,
    "cross_fit": True,
    "outer_folds": 5,
    "learner_folds": 2,
    "simultaneous_intervals": False,
    "variance_method": "ic",
    "stratify": True,
    "g_bounds": list(G_BOUNDS),
    "regimens": list(REGIMENS),
    # The sequential regressions both sides run, written as designs rather than in any one
    # package's formula language.
    "outcome_designs": [["W1", "W2"], ["W1", "W2", "L2"]],
    # The mechanism is supplied to both implementations rather than estimated by each, which
    # is what makes the paired verdict a statement about the recursion.  Letting each side
    # estimate it produced a comparison of two mechanism pipelines: lmtp's SL.glm ratio came
    # out shrunken, its intervals covered 0.75 to 0.91, and one RMSE bound missed its margin.
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
    return frame, {name: float(truth[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(STUDY, scenario, replicate))


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        # The law's own mechanism, given to both implementations.  R ``lmtp`` receives the
        # same probabilities as exact per-node density ratios, because it has no ``gform``
        # equivalent to be handed them through.  This is the ordinary rows' principle: a
        # paired comparison that also estimates the mechanism two different ways measures
        # two unrelated pipelines rather than the recursion it exists to compare.  The
        # sequential regressions stay misspecified on both sides, so targeting still has
        # work to do.
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=5,
        learner_folds=2,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _initials(result: Any) -> dict[str, float]:
    means = {
        f"ey_regimen[{label}]": float(np.mean(result.fits[label].steps[0].initial))
        for label in REGIMENS
    }
    means["ate_regimen[always vs never]"] = means["ey_regimen[always]"] - means["ey_regimen[never]"]
    means[f"ate_regimen[{RULE_LABEL} vs never]"] = (
        means[f"ey_regimen[{RULE_LABEL}]"] - means["ey_regimen[never]"]
    )
    return means


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return _rows_from_result(fit_cleverly(frame), truth, scenario, replicate)


def _rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    initials = _initials(result)
    rows: list[dict[str, Any]] = []
    for name in ESTIMANDS:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
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
                "initial_estimate": initials[name],
            }
        )
    return rows


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    sample = frame.copy()
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, _rows_from_result(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
