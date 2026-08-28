"""Canonical ordinary competing-risk longitudinal TMLE evidence study.

The paired comparison gives cleverly and pinned R ``lmtp`` the same sampled panels,
intervention assignments, exact treatment and censoring mechanisms, and working
regressions.  R fits each cause with the other cause declared through ``compete=``.
The independent property study lives in ``ltmle_competing_properties``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_competing as law
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import G_BOUNDS
from tests.studies.canonical_ltmle_crossfit import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 4_000
SEED = 20260826
SCENARIO = "censored_competing_risk_curve"

REGIMENS = law.REGIMEN_SPEC
REFERENCE = law.REGIMEN_REFERENCE
HORIZONS = law.HORIZONS
CAUSES = law.CAUSES

# ``continue_if_l2`` is identical to ``always`` at the first node.  The duplicated
# parameters remain exact-law identities rather than repeated-sampling cells.
MEAN_NAMES = tuple(
    f"cif_regimen[{label}, {cause} @ t={horizon}]"
    for cause in CAUSES
    for label, horizon in (
        ("never", 1),
        ("never", 2),
        ("always", 1),
        ("always", 2),
        ("continue_if_l2", 2),
    )
)
CONTRAST_NAMES = tuple(
    f"ate_regimen[{label} vs never, {cause} @ t={horizon}]"
    for cause in CAUSES
    for label, horizon in (("always", 1), ("always", 2), ("continue_if_l2", 2))
)
ESTIMANDS = (*MEAN_NAMES, *CONTRAST_NAMES)
PROPERTY_LABELS = ("relapse_dynamic_t2", "death_static_t2")


def property_cells(*, crossfit: bool) -> dict[str, tuple[str, ...]]:
    """The property cells both competing-risk studies declare, cross-fitting aside.

    Public, unlike the equivalent in every other study module, because the cross-fitted
    study calls it: the two rows differ in one flag and nothing else, so a second copy of
    this table would be a place for the two to disagree about what they measured.  The same
    goes for :func:`manifest_configuration` and :func:`rows_from_result` below.
    """
    cells = {
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
            f"{label}__n_{size}" for label in PROPERTY_LABELS for size in (4000, 8000, 32000)
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
        "competing_risk_recursion_necessity": tuple(
            f"{cause}_always_t2__{arm}"
            for cause in CAUSES
            for arm in ("all_cause", "cause_specific_control")
        ),
    }
    if crossfit:
        cells["crossfit_overfitting"] = (
            "cross_fitted_competing_ltmle",
            "in_sample_control",
        )
    return cells


REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
}


def manifest_configuration(*, crossfit: bool) -> dict[str, Any]:
    """What the manifest records about how either row was fitted."""
    return {
        "construction": "fold_specific_cross_fit" if crossfit else "ordinary",
        "outcome_kind": "competing_absorbing_events",
        "horizon_mode": "all_prefixes",
        "r_survival_outcome": True,
        "cross_fit": crossfit,
        "outer_folds": 5 if crossfit else 1,
        "learner_folds": 2,
        "simultaneous_intervals": False,
        "variance_method": "ic",
        "stratify": True,
        "g_bounds": list(G_BOUNDS),
        "horizons": list(HORIZONS),
        "causes": list(CAUSES),
        "regimens": list(REGIMENS),
        "outcome_designs": [["intercept"], ["intercept"]],
        "mechanism": "supplied_from_the_law_to_both",
        "reference_density_ratios": "exact_per_node",
        "reference_competing_event": "the_other_cause",
    }


STUDY = StudyRecord(
    name="ordinary competing-risk longitudinal TMLE",
    slug="canonical-ltmle-competing",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_ltmle_competing",
    document=(
        "docs/technical-reference/method-evidence/ordinary-competing-risk-longitudinal-tmle.md"
    ),
    anchor="ordinary-competing-risk-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260828,
    margins=Margins(),
    implementation="cleverly-competing-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_ltmle_competing.py",
        "tests/studies/ltmle_competing_properties.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/discrete_law_competing.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_competing_adapter.R",
        "tests/canonical/lmtp_ltmle_competing/run_study.R",
    ),
    runner_module="tests.studies.canonical_ltmle_competing",
    properties_module="tests.studies.ltmle_competing_properties",
    property_cells=property_cells(crossfit=False),
)

CONFIGURATION = manifest_configuration(crossfit=False)


class KnownCompetingMechanism(BaseEstimator):
    """The exact treatment or censoring law, keyed by the realized design."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownCompetingMechanism:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        index = np.rint(np.nan_to_num(matrix)).astype(int)
        if self.kind == "treatment" and matrix.shape[1] == 1:
            probability = law.G1[index[:, 0]]
        elif self.kind == "treatment" and matrix.shape[1] == 3:
            probability = law.G2[index[:, 0], index[:, 2], index[:, 1]]
        elif self.kind == "censoring" and matrix.shape[1] == 2:
            probability = law.C1[index[:, 0], index[:, 1]]
        elif self.kind == "censoring" and matrix.shape[1] == 4:
            probability = law.C2[index[:, 0], index[:, 2], index[:, 1], index[:, 3]]
        else:  # pragma: no cover - a changed design is a study-contract failure
            raise ValueError(f"unexpected {self.kind} mechanism design {matrix.shape}")
        return np.column_stack([1.0 - probability, probability])


_LAW_FRAME = law.frame()


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one panel from the finite-support competing-risk law."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    rows = np.random.default_rng(seed).integers(0, law.N, size=n)
    frame = _LAW_FRAME.iloc[rows].reset_index(drop=True)
    return frame, {name: float(law.TRUTH[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, *, n_folds: int = 1) -> Any:
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=DummyClassifier(strategy="prior"),
        pseudo_learner=DummyRegressor(strategy="mean"),
        treatment_learner=KnownCompetingMechanism("treatment"),
        censoring_learner=KnownCompetingMechanism("censoring"),
        n_folds=n_folds,
        learner_folds=2,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=law.outcome_columns(),
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _initials(result: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in MEAN_NAMES:
        inside = name[len("cif_regimen[") : -1]
        label_cause, horizon_text = inside.rsplit(" @ t=", 1)
        fit = result.fits[f"{label_cause} @ t={int(horizon_text)}"]
        values[name] = float(np.mean(fit.steps[0].initial))
    for name in CONTRAST_NAMES:
        inside = name[len("ate_regimen[") : -1]
        comparison_cause, horizon_text = inside.rsplit(" @ t=", 1)
        comparison, cause = comparison_cause.rsplit(", ", 1)
        left, right = comparison.split(" vs ")
        horizon = int(horizon_text)
        values[name] = (
            values[f"cif_regimen[{left}, {cause} @ t={horizon}]"]
            - values[f"cif_regimen[{right}, {cause} @ t={horizon}]"]
        )
    return values


def rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    *,
    study: StudyRecord = STUDY,
) -> list[dict[str, Any]]:
    initials = _initials(result)
    rows: list[dict[str, Any]] = []
    for name in ESTIMANDS:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": study.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(result.folds.assignment),
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


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return rows_from_result(fit_cleverly(frame), truth, scenario, replicate)


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
    return sample, truths, rows_from_result(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
