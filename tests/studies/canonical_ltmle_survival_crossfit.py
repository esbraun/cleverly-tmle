"""Canonical cross-fitted survival-curve longitudinal TMLE evidence study.

The registered row has no numeric comparator because the pinned ``lmtp`` audit failed its
cross-fitted truth-coverage gates. Statistical properties are checked independently in
``tests.studies.ltmle_survival_crossfit_properties`` against the finite-support survival law.
The retained R runner receives the same panels and exact fold assignment for source audits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import RULE_LABEL, make_longitudinal_survival, rule_arm_at_node_two
from cleverly.datasets.longitudinal import survival_truth
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism, QuasiBinomialGLM
from tests.studies.canonical_ltmle_crossfit import (
    G_BOUNDS,
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.canonical_ltmle_survival import dynamic_survival_truth
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20260825
SCENARIO = "censored_survival_curve"

REGIMENS: dict[str, Any] = {
    "never": 0,
    "always": 1,
    RULE_LABEL: (1, lambda history: rule_arm_at_node_two(history["L2"])),
}
REFERENCE = "never"
HORIZONS = (1, 2)

# The dynamic plan is identical to always at t=1.  The duplicate level and contrast are
# exact structural identities, not additional repeated-sampling evidence, so the study
# reports them in a fast test rather than counting them as primary cells.
MEAN_NAMES = (
    "risk_regimen[never @ t=1]",
    "risk_regimen[never @ t=2]",
    "risk_regimen[always @ t=1]",
    "risk_regimen[always @ t=2]",
    f"risk_regimen[{RULE_LABEL} @ t=2]",
)
CONTRAST_NAMES = (
    "ate_regimen[always vs never @ t=1]",
    "ate_regimen[always vs never @ t=2]",
    f"ate_regimen[{RULE_LABEL} vs never @ t=2]",
)
ESTIMANDS = (*MEAN_NAMES, *CONTRAST_NAMES)

PROPERTY_LABELS = ("static_t1", "static_t2", "dynamic_t2")

STUDY = StudyRecord(
    name="cross-fitted survival-curve longitudinal TMLE",
    slug="canonical-ltmle-survival-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_ltmle_survival",
    document="docs/technical-reference/method-evidence.md",
    anchor="cross-fitted-survival-curve-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-ltmle-survival",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_ltmle_survival_crossfit.py",
        "tests/studies/ltmle_survival_crossfit_properties.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_ltmle_survival.py",
        "tests/discrete_law_survival.py",
        "src/cleverly/datasets/longitudinal.py",
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
        "tests/canonical/lmtp_ltmle_survival/run_study.R",
    ),
    runner_module="tests.studies.canonical_ltmle_survival_crossfit",
    properties_module="tests.studies.ltmle_survival_crossfit_properties",
    property_cells={
        "double_robustness": tuple(
            f"{label}__{configuration}"
            for label in PROPERTY_LABELS
            for configuration in (
                "both_correct",
                "outcome_correct",
                "mechanism_correct",
                "both_wrong",
            )
        ),
        "root_n_and_efficiency": tuple(
            f"{label}__n_{size}" for label in PROPERTY_LABELS for size in (1000, 2000, 8000)
        ),
        "root_n_rate": tuple(
            f"{label}__{statistic}"
            for label in PROPERTY_LABELS
            for statistic in ("empirical_sd", "reported_se")
        ),
        "interval_calibration": tuple(
            f"{label}__{cell}"
            for label in PROPERTY_LABELS
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
        "type_i_error": tuple(f"{label}__sharp_null" for label in PROPERTY_LABELS),
        "power": tuple(f"{label}__alternative" for label in PROPERTY_LABELS),
        "targeting_necessity": tuple(
            f"{label}__{arm}" for label in PROPERTY_LABELS for arm in ("targeted", "untargeted")
        ),
        "survival_recursion_necessity": (
            "always_t2__survival",
            "always_t2__survivor_only",
        ),
        "crossfit_overfitting": (
            "cross_fitted_survival_ltmle",
            "in_sample_control",
        ),
    },
)

#: Inert while this study declares no comparator.  See
#: :data:`tests.studies.canonical_ltmle_crossfit.REFERENCE_METADATA`.
REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "fold_specific_cross_fit",
    "outcome_kind": "absorbing_event",
    "horizon_mode": "all_prefixes",
    "r_survival_outcome": True,
    "cross_fit": True,
    "outer_folds": 5,
    "learner_folds": 2,
    "simultaneous_intervals": False,
    "variance_method": "ic",
    "stratify": True,
    "g_bounds": list(G_BOUNDS),
    "horizons": list(HORIZONS),
    "regimens": list(REGIMENS),
    "outcome_designs": [["W1", "W2"], ["W1", "W2", "L2"]],
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node",
}


def truths() -> dict[str, float]:
    """Every primary parameter, independently of either fitted implementation."""
    values = {
        f"risk_regimen[{label} @ t={horizon}]": survival_truth(arm, arm, horizon)
        for label, arm in (("never", 0.0), ("always", 1.0))
        for horizon in HORIZONS
    }
    values[f"risk_regimen[{RULE_LABEL} @ t=2]"] = dynamic_survival_truth()
    for label, horizon in (("always", 1), ("always", 2), (RULE_LABEL, 2)):
        values[f"ate_regimen[{label} vs never @ t={horizon}]"] = (
            values[f"risk_regimen[{label} @ t={horizon}]"]
            - values[f"risk_regimen[never @ t={horizon}]"]
        )
    return {name: float(values[name]) for name in ESTIMANDS}


TRUTH = truths()


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, _ = make_longitudinal_survival(n=n, seed=seed, censoring=True, backend="pandas")
    return frame, dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(STUDY, scenario, replicate))


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        # The law's own mechanism, given to both implementations.  See
        # :func:`tests.studies.canonical_ltmle_crossfit.fit_cleverly`, which carries the
        # argument: a paired comparison that also estimates the mechanism two different ways
        # measures two pipelines rather than the recursion.
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
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _initials(result: Any) -> dict[str, float]:
    means: dict[str, float] = {}
    for name in MEAN_NAMES:
        inner = name[len("risk_regimen[") : -1]
        label, horizon_text = inner.rsplit(" @ t=", 1)
        fit = result.fits[f"{label} @ t={int(horizon_text)}"]
        means[name] = float(np.mean(fit.steps[0].initial))
    for name in CONTRAST_NAMES:
        inner = name[len("ate_regimen[") : -1]
        comparison, horizon_text = inner.rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        horizon = int(horizon_text)
        means[name] = (
            means[f"risk_regimen[{left} @ t={horizon}]"]
            - means[f"risk_regimen[{right} @ t={horizon}]"]
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
    truths_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths_rows, _rows_from_result(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths_frame = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths_frame, estimates.loc[:, list(REPLICATE_COLUMNS)]
