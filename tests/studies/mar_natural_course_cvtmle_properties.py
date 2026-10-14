"""Repeated-sampling properties for stacked MAR natural-course CV-TMLE."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier

from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_mar_natural_course_cvtmle import (
    NUISANCE_BOUND,
    STUDY,
    TRUTH,
    fit_cleverly,
    flexible_outcome_learner,
    flexible_response_learner,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    ReplicationSpec,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
    replication_payloads,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.missing_outcome_study_helpers import (
    NaturalCourseLaw,
    natural_course_law,
    sample_discrete,
)

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
CALIBRATION_REPLICATES = 2_000
CALIBRATION_N = 2_000
OVERFIT_REPLICATES = 400
OVERFIT_N = 500
SHRUNKEN_SE_FACTOR = 0.70
TARGET = "ey_obs"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
NOISE_COLUMNS = tuple(f"N{index}" for index in range(1, 9))


def _with_noise(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add prognostically irrelevant continuous features without changing the exact truth."""
    out = frame.copy()
    rng = np.random.default_rng(np.random.SeedSequence((seed, 20261111)))
    for name in NOISE_COLUMNS:
        out[name] = rng.normal(size=len(out))
    return out


def _fit_union(frame: pd.DataFrame, configuration: str) -> Any:
    law = natural_course_law(configuration)
    return fit_cleverly(
        frame,
        outcome_learner=OracleOutcome(law),
        missingness_learner=OracleMissingness(law),
    )


def _fully_grown_outcome() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(min_samples_leaf=1, random_state=23)


_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, which is a function of the key alone."""
    property_name, _, _, n, _, seed, _ = key[0]
    frame = sample_discrete(mar.PROBS, n, seed)
    return _with_noise(frame, seed) if property_name == "crossfit_overfitting" else frame


def _cells_of(key: _Key) -> list[str]:
    """The ``property/cell`` names the key's rows fill."""
    property_name, cell = key[0][:2]
    names = [f"{property_name}/{cell}"]
    if property_name == "interval_calibration":
        names.append(f"{property_name}/ey_obs__shrunken_se_control")
    return names


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    (payload,) = key
    property_name, cell, replicate, n, requested, _seed, configuration = payload
    role = "control" if configuration in {"both_wrong", "in_sample"} else "positive"

    if property_name == "double_robustness":
        result = _fit_union(frame, configuration)
    elif property_name == "interval_calibration":
        result = fit_cleverly(
            frame,
            outcome_learner=flexible_outcome_learner(),
            missingness_learner=flexible_response_learner(),
        )
    elif property_name == "crossfit_overfitting":
        result = fit_cleverly(
            frame,
            outcome_learner=_fully_grown_outcome(),
            # The law reads ``W`` from the first covariate column and ignores the noise.
            missingness_learner=OracleMissingness(NaturalCourseLaw()),
            cross_fit=configuration != "in_sample",
            adjustment=("W", *NOISE_COLUMNS),
        )
    else:  # pragma: no cover - declaration guard
        raise KeyError(property_name)

    estimate = result[TARGET]
    rows = [
        replicate_row(
            property_name=property_name,
            cell=cell,
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=TRUTH,
            estimate=estimate,
            alpha=STUDY.margins.alpha,
        )
    ]
    if property_name == "interval_calibration":
        rows.append(
            control_row(
                property_name=property_name,
                cell="ey_obs__shrunken_se_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=TRUTH,
                estimate=float(estimate.psi),
                standard_error=SHRUNKEN_SE_FACTOR * float(estimate.std_error),
                critical=CRITICAL,
            )
        )
    return rows


def _specs() -> tuple[list[ReplicationSpec], list[ReplicationSpec]]:
    """Return the independent specifications and the paired overfitting arms."""
    independent = [
        ReplicationSpec(
            "double_robustness",
            configuration,
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
        )
        for configuration in ("outcome_correct", "response_correct", "both_wrong")
    ]
    independent.append(
        ReplicationSpec(
            "interval_calibration",
            "ey_obs__flexible_learning",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "flexible_learning",
            seed_key="flexible_learning",
        )
    )
    # Both arms deliberately share every draw; only the sample-splitting policy changes.
    paired = [
        ReplicationSpec(
            "crossfit_overfitting",
            cell,
            OVERFIT_N,
            OVERFIT_REPLICATES,
            configuration,
            seed_key="paired",
        )
        for cell, configuration in (
            ("stacked_mar_natural_course_cvtmle", "cross_fit"),
            ("in_sample_control", "in_sample"),
        )
    ]
    return independent, paired


def _payloads(
    replicates: int | None = None,
) -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    """Expand the specifications, optionally truncated to ``replicates`` per cell.

    The paired arms alternate replicate by replicate, so each draw's two fits sit together.
    """
    independent, paired = _specs()
    if replicates is not None:
        independent = [replace(spec, replicates=replicates) for spec in independent]
        paired = [replace(spec, replicates=replicates) for spec in paired]
    cross_fit, in_sample = (replication_payloads(STUDY, [spec]) for spec in paired)
    alternating = [payload for pair in zip(cross_fit, in_sample, strict=True) for payload in pair]
    return replication_payloads(STUDY, independent) + alternating


def _fit(payloads: list[_Key], *, n_jobs: int, label: str, checkpoint: Any) -> pd.DataFrame:
    outcomes = fit_properties(
        payloads,
        _fit_replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        label=label,
        draw=_draw,
        checkpoint=checkpoint,
    )
    return pd.DataFrame([row for result in outcomes for row in result])


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return _fit(_payloads(), n_jobs=n_jobs, label="properties", checkpoint="ambient")


def generate_smoke_property_rows(*, n_jobs: int = 1) -> pd.DataFrame:
    """Fit one declared replication of every property cell without summarizing it."""
    return _fit(_payloads(replicates=1), n_jobs=n_jobs, label="properties/smoke", checkpoint=None)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        rate_labels=(),
    )
    calibration_verdicts(
        summary,
        margins=STUDY.margins,
        positive_suffix="flexible_learning",
    )
    crossfit_overfitting_verdicts(
        summary,
        rows,
        STUDY,
        positive_cell="stacked_mar_natural_course_cvtmle",
    )
    return finish(summary, rates)


def declared_settings() -> dict[str, Any]:
    """Expose result-determining constants for the generated evidence document."""
    return {
        "double_robust_replicates": DOUBLE_ROBUST_REPLICATES,
        "double_robust_n": DOUBLE_ROBUST_N,
        "calibration_replicates": CALIBRATION_REPLICATES,
        "calibration_n": CALIBRATION_N,
        "overfit_replicates": OVERFIT_REPLICATES,
        "overfit_n": OVERFIT_N,
        "shrunken_se_factor": SHRUNKEN_SE_FACTOR,
        "nuisance_bound": NUISANCE_BOUND,
    }
