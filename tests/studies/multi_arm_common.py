"""Shared law, estimands, learners, and row adapters for multi-arm evidence.

The registered rows deliberately remain method-specific.  This module owns the pieces that
must not drift between them: all methods see the same labelled three-arm law, use the same
reference arm, report the same derived contrasts, and draw from the seed stream declared by
their own :class:`~tests.studies.evidence.registry.StudyRecord`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from cleverly._typing import EstimandName
from cleverly.datasets import MultiArmDGP, multi_arm_dgp
from cleverly.utils.bounds import expit
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

LABELS = ("high", "low", "medium")
REFERENCE = "high"
MEAN_ESTIMANDS: tuple[EstimandName, ...] = (
    "ey[high]",
    "ey[low]",
    "ey[medium]",
    "ate[low vs high]",
    "ate[medium vs high]",
)
RATIO_ESTIMANDS: tuple[EstimandName, ...] = (
    "rr[low vs high]",
    "rr[medium vs high]",
    "or[low vs high]",
    "or[medium vs high]",
)
ALL_ESTIMANDS: tuple[EstimandName, ...] = (*MEAN_ESTIMANDS, *RATIO_ESTIMANDS)
G_BOUNDS = (0.025, 0.975)


def law(*, effect: float = 0.6) -> MultiArmDGP:
    """The shared binary law, linear in sorted arm code and saturated in Python.

    The archived ``ctmle3`` snapshot documents categorical treatment but its old GLM
    adapter drops a factor column when predicting counterfactual tasks.  Supplying the same
    treatment as numeric codes avoids that adapter defect.  The law is therefore linear in
    those codes, while Cleverly still receives the labelled treatment and its saturated
    indicator design.  Both nuisance models are exactly specified and the label/code
    distinction remains exercised.
    """
    base = multi_arm_dgp(family="binomial")

    def arm_logits(w: np.ndarray) -> np.ndarray:
        # Keep this registered law inside the estimator's declared 2.5% truncation
        # region at all practically observable covariate values.  The package's generic
        # multi-arm fixture deliberately has much stronger tails; using it here made a
        # cell labelled "treatment correct" study a truncated mechanism instead.  These
        # three linear logits remain a correctly specified, confounded multinomial model.
        return np.column_stack(
            (
                np.zeros(len(w)),
                0.25 * w[:, 0] - 0.15 * w[:, 1],
                -0.15 * w[:, 0] + 0.25 * w[:, 1],
            )
        )

    def outcome_mean(w: np.ndarray, arm: int) -> np.ndarray:
        code = LABELS.index(base.labels[arm])
        return expit(-0.5 + effect * code + 0.6 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2])

    return replace(
        base,
        name="multi_arm_evidence_binary",
        arm_logits=arm_logits,
        outcome_mean=outcome_mean,
    )


def truth_for(process: MultiArmDGP) -> dict[str, float]:
    """Arm means, reference-arm differences, risk ratios, and odds ratios."""
    truth = process.truth(reference=REFERENCE)
    reference = truth[f"ey[{REFERENCE}]"]
    for label in LABELS:
        if label == REFERENCE:
            continue
        mean = truth[f"ey[{label}]"]
        truth[f"rr[{label} vs {REFERENCE}]"] = mean / reference
        truth[f"or[{label} vs {REFERENCE}]"] = (mean / (1.0 - mean)) / (
            reference / (1.0 - reference)
        )
    return truth


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw the shared law from an explicit seed for manifest-seed auditing."""
    del scenario
    process = law()
    frame, _ = process.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(process)


def draw_for(
    record: StudyRecord, scenario: str, n: int, replicate: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw from ``record``'s seed stream rather than from another row's stream."""
    return draw_replicate(record, draw_from_seed, scenario, n, replicate)


def rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    *,
    implementation: str,
    scenario: str,
    replicate: int,
    n: int,
    estimands: Sequence[str],
) -> list[dict[str, Any]]:
    """Convert any multi-arm result into the common primary-evidence schema."""
    rows: list[dict[str, Any]] = []
    initial = result.repeats[0].nuisance.outcome.arms
    initial_means = {
        result.data.arm_label(code): float(np.mean(values)) for code, values in initial.items()
    }
    for name in estimands:
        estimate = result[name]
        target = float(truth[name])
        low, high = estimate.ci
        ratio = estimate.scale == "ratio"
        if name.startswith("ey["):
            initial_estimate = initial_means[name[3:-1]]
        else:
            stem, _, content = name.partition("[")
            label, reference = content[:-1].split(" vs ")
            left = initial_means[label]
            right = initial_means[reference]
            if stem == "ate":
                initial_estimate = left - right
            elif stem == "rr":
                initial_estimate = left / right
            elif stem == "or":
                initial_estimate = (left / (1.0 - left)) / (right / (1.0 - right))
            else:
                raise ValueError(f"no untargeted plug-in is defined for {name!r}")
        rows.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": target,
                "estimate": float(estimate.psi),
                "inference_estimate": (
                    float(estimate.log_psi)
                    if ratio and estimate.log_psi is not None
                    else float(estimate.psi)
                ),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if ratio else "identity",
                "covered": int(low <= target <= high),
                "initial_estimate": initial_estimate,
            }
        )
    return rows


def cleverly_rows(
    record: StudyRecord,
    fitter: Callable[[pd.DataFrame, str], Any],
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Refit one subject replication through the registered row schema."""
    return rows_from_result(
        fitter(frame, scenario),
        truth,
        implementation=record.implementation,
        scenario=scenario,
        replicate=replicate,
        n=len(frame),
        estimands=record.scenarios[scenario],
    )


def draw_and_fit(
    record: StudyRecord,
    fitter: Callable[[pd.DataFrame, str], Any],
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    include_samples: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a method-specific fitter while sharing sampling, packing, and row conversion."""

    def replicate(
        payload: tuple[str, int, int],
    ) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
        scenario, index, size = payload
        frame, truth = draw_for(record, scenario, size, index)
        result = fitter(frame, scenario)
        sample = frame.copy()
        sample["A_code"] = pd.Categorical(sample["A"], categories=LABELS).codes
        sample.insert(0, "replicate", index)
        sample.insert(0, "scenario", scenario)
        truth_row = {
            "scenario": scenario,
            "replicate": index,
            **{f"truth_{name}": value for name, value in truth.items()},
        }
        rows = rows_from_result(
            result,
            truth,
            implementation=record.implementation,
            scenario=scenario,
            replicate=index,
            n=size,
            estimands=record.scenarios[scenario],
        )
        return sample, truth_row, rows

    payloads = [
        ((scenario, replicate, n),)
        for scenario in record.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(replicate, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for _, _, fitted in outcomes for row in fitted]).loc[
        :, list(REPLICATE_COLUMNS)
    ]
    if not include_samples:
        return rows
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    return samples, truths, rows


class OracleMultiTreatment(BaseEstimator):
    """The shared law's exact multinomial treatment mechanism."""

    def __init__(self, process: MultiArmDGP) -> None:
        self.process = process

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> OracleMultiTreatment:
        del design, target, sample_weight
        self.classes_ = np.arange(float(self.process.n_arms))
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        probabilities = self.process.probabilities(values[:, : self.process.n_latent])
        # The process axis follows its declared labels; estimator codes follow sorted labels.
        return np.column_stack(
            [probabilities[:, self.process.labels.index(label)] for label in LABELS]
        )


class OracleMultiOutcome(BaseEstimator):
    """The shared law's exact outcome regression on the estimator's indicator design."""

    def __init__(self, process: MultiArmDGP) -> None:
        self.process = process

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> OracleMultiOutcome:
        del design, target, sample_weight
        return self

    def _mean(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        indicators = values[:, : self.process.n_arms - 1]
        code = np.where(indicators.any(axis=1), indicators.argmax(axis=1) + 1, 0)
        latent = values[:, self.process.n_arms - 1 :]
        means = np.empty(len(values), dtype=float)
        for arm_code, label in enumerate(LABELS):
            selected = code == arm_code
            source_arm = self.process.labels.index(label)
            means[selected] = self.process.outcome_mean(latent[selected], source_arm)
        return means

    def predict(self, design: Any) -> np.ndarray:
        return self._mean(design)

    def predict_proba(self, design: Any) -> np.ndarray:
        mean = self._mean(design)
        return np.column_stack((1.0 - mean, mean))


__all__ = [
    "ALL_ESTIMANDS",
    "G_BOUNDS",
    "LABELS",
    "MEAN_ESTIMANDS",
    "RATIO_ESTIMANDS",
    "REFERENCE",
    "OracleMultiOutcome",
    "OracleMultiTreatment",
    "cleverly_rows",
    "draw_and_fit",
    "draw_for",
    "draw_from_seed",
    "law",
    "rows_from_result",
    "truth_for",
]
