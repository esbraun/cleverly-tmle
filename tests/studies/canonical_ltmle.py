"""Canonical ordinary end-of-study longitudinal TMLE method-evidence study.

The paired comparison deliberately supplies the treatment and censoring mechanisms from
the law to both implementations.  It therefore compares the sequential regressions,
targeting, influence curves and correlated regimen contrasts rather than two unrelated
mechanism-fitting pipelines.  Statistical properties are checked independently in
``tests.studies.ltmle_properties`` against the finite-support longitudinal law.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator

from cleverly.datasets import RULE_LABEL, make_longitudinal, rule_arm_at_node_two
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

LTMLE_VERSION = "1.3-0"
LTMLE_SOURCE_COMMIT = "338c029dae9692ef20714125773da7037688993b"
LTMLE_TARBALL_SHA256 = "fb31d0dd6ab81687b81f3279b414c21e91c655e10aac12f73fc6723efd848aad"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20260822
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
    name="ordinary end-of-study longitudinal TMLE",
    slug="canonical-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "ltmle",
    document="docs/technical-reference/method-evidence/ordinary-end-of-study-longitudinal-tmle.md",
    anchor="ordinary-end-of-study-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="ltmle",
    modules=(
        "tests/studies/canonical_ltmle.py",
        "tests/studies/ltmle_properties.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_ltmle",
    properties_module="tests.studies.ltmle_properties",
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
            for size in (500, 2000, 8000)
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
    },
)

REFERENCE_METADATA = {
    "ltmle_version": LTMLE_VERSION,
    "ltmle_source_commit": LTMLE_SOURCE_COMMIT,
    "ltmle_tarball_sha256": LTMLE_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "ordinary",
    "outcome_kind": "end_of_study",
    "horizon_mode": "terminal_only",
    "r_survival_outcome": False,
    "cross_fit": False,
    "simultaneous_intervals": False,
    "variance_method": "ic",
    "stratify": True,
    "g_bounds": list(G_BOUNDS),
    "regimens": list(REGIMENS),
    "q_formulas": ["Q.kplus1 ~ W1 + W2", "Q.kplus1 ~ W1 + W2 + L2"],
}


class QuasiBinomialGLM(BaseEstimator):
    """Small scikit-compatible unpenalized quasibinomial IRLS learner.

    R ``ltmle`` uses ``glm(..., family=quasibinomial())`` for both the binary final
    outcome and fractional earlier pseudo-outcomes.  Scikit-learn's logistic classifier
    refuses fractional targets, so the canonical study carries the corresponding score
    solver rather than silently comparing different regression families.
    """

    def __init__(self, *, max_iter: int = 100, tol: float = 1e-10) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> QuasiBinomialGLM:
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = (
            np.ones_like(target)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        design = np.column_stack([np.ones(len(matrix)), matrix])
        mean = float(np.average(target, weights=weights))
        coefficient = np.zeros(design.shape[1], dtype=float)
        coefficient[0] = math.log(np.clip(mean, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - mean, 1e-8, 1.0))
        for _ in range(self.max_iter):
            fitted = expit(design @ coefficient)
            variance = np.clip(fitted * (1.0 - fitted), 1e-10, None)
            working = design @ coefficient + (target - fitted) / variance
            root_weight = np.sqrt(weights * variance)
            updated = np.linalg.lstsq(
                design * root_weight[:, None], working * root_weight, rcond=None
            )[0]
            if np.max(np.abs(updated - coefficient)) <= self.tol:
                coefficient = updated
                break
            coefficient = updated
        else:
            raise RuntimeError("quasibinomial IRLS did not converge")
        self.coef_ = coefficient[1:][None, :]
        self.intercept_ = coefficient[:1]
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return expit(self.intercept_[0] + matrix @ self.coef_[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        probability = self.predict(X)
        return np.column_stack([1.0 - probability, probability])


class KnownLongitudinalMechanism(BaseEstimator):
    """The generating treatment or censoring probabilities, keyed by design shape.

    Shared by the end-of-study study here, by the two survival studies, and by the
    overfitting cells of both cross-fitted property studies.  Reading a column by *position*
    is safe across all of them because of a contract rather than a coincidence.
    :meth:`~cleverly.longitudinal.LongitudinalData.history_design` builds a mechanism's
    conditioning set as ``[W, L_1, ..., L_t]`` followed by one block per earlier treatment,
    plus the current one for a censoring model.  **An outcome node never enters it**, so the
    survival panel's ``Y1`` cannot shift ``L2`` or ``A1`` out of the position this reads them
    from, and both panels present ``[W1, W2, L2, A1]`` at the second treatment node and
    ``[W1, W2, L2, A1, A2]`` at the second censoring node.

    A width this does not recognise raises rather than guessing.  A *reordering* within a
    width would not, and it needs its own guard, because a permutation of two equally wide
    blocks changes every probability and leaves the shape alone.

    In the paired studies the guard comes free: the outcome regression there is a ``glm``
    against a law with a ``tanh`` term in it, so a mechanism read off the wrong columns biases
    that side and the agreement with R breaks loudly.  **The cross-fitted overfitting cells
    have no such guard.** They run against ``make_longitudinal`` rather than the discrete law,
    and no registered comparison fits that pair, so nothing downstream would notice.  For
    those, ``tests/unit/test_ltmle_crossfit_method_study.py`` checks this class against the
    generating probabilities directly, on both panels and at both nodes; a swap of any two
    design columns moves them by between 0.08 and 0.64, against a tolerance of ``1e-12``.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownLongitudinalMechanism:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if self.kind == "treatment" and matrix.shape[1] == 2:
            probability = expit(0.3 * matrix[:, 0] - 0.4 * matrix[:, 1])
        elif self.kind == "treatment" and matrix.shape[1] == 4:
            probability = expit(0.5 * matrix[:, 2] + 0.6 * matrix[:, 3] - 0.2 * matrix[:, 1])
        elif self.kind == "censoring" and matrix.shape[1] == 3:
            probability = expit(2.2 + 0.3 * matrix[:, 0] - 0.3 * matrix[:, 2])
        elif self.kind == "censoring" and matrix.shape[1] == 5:
            probability = expit(2.4 + 0.2 * matrix[:, 2])
        else:  # pragma: no cover - a changed longitudinal design is a study-contract failure
            raise ValueError(f"unexpected {self.kind} mechanism design {matrix.shape}")
        return np.column_stack([1.0 - probability, probability])


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
    return frame, {name: float(truth[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=1,
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


def untargeted(frame: pd.DataFrame, label: str) -> float:
    r"""The sequential-regression plug-in for one plan on the comparison law, unfluctuated.

    The same two follower-stratified quasibinomial regressions both implementations run --
    ``Q.kplus1 ~ W1 + W2 + L2`` at the outcome node and ``Q.kplus1 ~ W1 + W2`` at the earlier
    one -- carried back and averaged, with the update in between left out.

    It exists because of what the paired comparison can and cannot see.  ``initial_estimate``
    is the earlier node's regression of the *already targeted* later node, in R
    (``fit$Q[[1]]`` regresses the updated ``Q.kplus1``) as much as here, so the published
    displacement measures the final fluctuation and not the whole targeting step.  This is the
    whole of it, which is what ``tests/e2e/test_ltmle_targeting_slow.py`` needs to state how
    much of the agreement between the two implementations the fluctuation is responsible for.
    """
    plan = REGIMENS[label]
    baseline = frame[["W1", "W2"]].to_numpy(dtype=float)
    l2 = np.nan_to_num(frame["L2"].to_numpy(dtype=float))
    history = np.column_stack([baseline, l2])
    if np.ndim(plan) == 0:
        first = second = np.full(len(frame), float(plan))
    else:
        first = np.full(len(frame), float(plan[0]))
        second = np.asarray(plan[1]({"L2": l2}), dtype=float)
    followed_one = (frame["C1"].to_numpy() == 1.0) & (frame["A1"].to_numpy() == first)
    followed_two = (
        followed_one & (frame["C2"].to_numpy() == 1.0) & (frame["A2"].to_numpy() == second)
    )
    later = QuasiBinomialGLM().fit(history[followed_two], frame["Y"].to_numpy()[followed_two])
    carried = later.predict(history)
    earlier = QuasiBinomialGLM().fit(baseline[followed_one], carried[followed_one])
    return float(np.mean(earlier.predict(baseline)))


def untargeted_estimands(frame: pd.DataFrame) -> dict[str, float]:
    """:func:`untargeted` for every reported estimand, contrasts included."""
    means = {f"ey_regimen[{label}]": untargeted(frame, label) for label in REGIMENS}
    for name in CONTRAST_NAMES:
        left, right = name[len("ate_regimen[") : -1].split(" vs ")
        means[name] = means[f"ey_regimen[{left}]"] - means[f"ey_regimen[{right}]"]
    return means


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
    result = fit_cleverly(frame)
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
                "n": len(frame),
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
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]
