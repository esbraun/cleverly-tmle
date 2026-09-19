"""Repeated-sampling properties for stacked CV-TMLE of arm-indexed means under MAR.

Four families, each fitted on the four laws of :mod:`tests.studies.mar_arm_indexed_laws`:

=========================  ===========================================================
family                     what it checks
=========================  ===========================================================
``interval_calibration``   coverage and SE calibration of every reported estimand with
                           the primary trees, and a shrunken-SE control of each
``simultaneous_coverage``  joint coverage of the multiplier band over each law's
                           reported estimands, and the same fits' pointwise intervals
                           read jointly as a control
``mar_robustness``         each ATE contrast with only ``Q``, only ``g``, or only
                           ``pi`` misspecified, and a control with ``Q`` and ``pi``
                           both misspecified
``crossfit_overfitting``   one ATE contrast per law with a fully grown outcome tree and
                           eight noise covariates, cross-fitted and in sample on the
                           same draws
=========================  ===========================================================

A ratio estimand's rows are on the log scale, the scale its interval is built on, so each
calibration statistic compares a standard error with the spread it describes.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import mar_arm_indexed_laws as laws
from tests.studies.canonical_mar_arm_indexed_cvtmle import (
    NUISANCE_BOUND,
    OVERFIT_LABELS,
    ROBUSTNESS_CONFIGURATIONS,
    STUDY,
    fit_cleverly,
)
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    ReplicationSpec,
    control_row,
    replication_payloads,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    robustness_verdicts,
)
from tests.studies.evidence.registry import Margins

ROBUSTNESS_REPLICATES = 1_200
ROBUSTNESS_N = 2_000
CALIBRATION_REPLICATES = 2_000
CALIBRATION_N = 2_000
OVERFIT_REPLICATES = 400
OVERFIT_N = 500
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
NOISE_COLUMNS = tuple(f"N{index}" for index in range(1, 9))
#: The calibration cells' positive arm.  Not ``flexible_learning``, whose shared description
#: names the natural-course study's two nuisances and six-cell law.
POSITIVE_SUFFIX = "learned_nuisances"
CONTROL = "outcome_and_response_wrong"


def _with_noise(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add prognostically irrelevant continuous features without changing the exact truth."""
    out = frame.copy()
    rng = np.random.default_rng(np.random.SeedSequence((seed, 20261303)))
    for name in NOISE_COLUMNS:
        out[name] = rng.normal(size=len(out))
    return out


def _fully_grown_outcome(law: laws.Law) -> Any:
    if law.continuous:
        return DecisionTreeRegressor(min_samples_leaf=1, random_state=23)
    return DecisionTreeClassifier(min_samples_leaf=1, random_state=23)


def _inference_scale(name: str, truth: float, estimate: Any) -> tuple[float, float, float, float]:
    """``(truth, estimate, lower, upper)`` on the scale the estimand's interval is built on."""
    low, high = estimate.ci
    if laws.is_ratio(name):
        return math.log(truth), float(estimate.log_psi), math.log(low), math.log(high)
    return truth, float(estimate.psi), float(low), float(high)


def _row(
    *,
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: float,
    std_error: float,
    covered: bool,
    rejected: bool,
) -> dict[str, Any]:
    return {
        "property": property_name,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": estimate,
        "std_error": std_error,
        "covered": int(covered),
        "rejected": int(rejected),
    }


def _estimand_row(
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    name: str,
    truth: float,
    estimate: Any,
) -> dict[str, Any]:
    target, point, low, high = _inference_scale(name, truth, estimate)
    return _row(
        property_name=property_name,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=target,
        estimate=point,
        std_error=float(estimate.std_error),
        covered=low <= target <= high,
        rejected=float(estimate.pvalue) < STUDY.margins.alpha,
    )


def _joint_rows(
    law: laws.Law,
    result: Any,
    truth: dict[str, float],
    *,
    replicate: int,
    n: int,
    requested: int,
) -> list[dict[str, Any]]:
    """The band row and its pointwise control, from one fit's estimates.

    A joint cell has no scalar estimand, so its row records the max-t statistic instead:
    ``estimate`` is the largest standardized deviation from the truth over the law's
    estimands, ``truth`` is zero, and ``std_error`` is the critical value the intervals
    used.  Only ``covered`` and ``rejected`` carry a verdict.  ``covered`` reads the
    package's own band and the package's own pointwise intervals, and ``rejected`` is its
    complement.
    """
    bands = result.simultaneous
    if bands is None:  # pragma: no cover - the calibration fit declares a band
        raise AssertionError("the calibration fit reported no simultaneous band")
    names = laws.ESTIMANDS[law.key]
    if set(bands.bands) != set(names):
        raise AssertionError(f"the band covers {sorted(bands.bands)}, not {sorted(names)}")
    deviations = []
    for name in names:
        target, point, _, _ = _inference_scale(name, truth[name], result[name])
        deviations.append(abs(point - target) / float(result[name].std_error))
    statistic = float(max(deviations))
    band = all(bands.bands[name][0] <= truth[name] <= bands.bands[name][1] for name in names)
    pointwise = all(result[name].ci[0] <= truth[name] <= result[name].ci[1] for name in names)
    return [
        _row(
            property_name="simultaneous_coverage",
            cell=f"{law.key}__{kind}",
            role=role,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=0.0,
            estimate=statistic,
            std_error=critical,
            covered=covered,
            rejected=not covered,
        )
        for kind, role, critical, covered in (
            ("simultaneous_band", "positive", float(bands.critical_value), band),
            ("pointwise_joint_control", "control", CRITICAL, pointwise),
        )
    ]


def _robustness_learners(law: laws.Law, configuration: str) -> dict[str, Any]:
    wrong = laws.wrong_tables(law)
    mu = wrong["mu"] if configuration in {"only_outcome_wrong", CONTROL} else None
    g = wrong["g"] if configuration == "only_treatment_wrong" else None
    pi = wrong["pi"] if configuration in {"only_response_wrong", CONTROL} else None
    return {
        "outcome_learner": laws.LawOutcome(law, mu),
        "treatment_learner": laws.LawTreatment(law, g),
        "missingness_learner": laws.LawResponse(law, pi),
    }


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, cell, replicate, n, requested, seed, configuration = payload
    law = laws.LAWS[cell.split("_", 1)[0]]
    frame = laws.sample(law, n, seed)
    truth = laws.truths(law)

    if property_name == "interval_calibration":
        result = fit_cleverly(frame, law, simultaneous=True)
        rows = []
        for name in laws.ESTIMANDS[law.key]:
            label = laws.label(law, name)
            positive = _estimand_row(
                property_name,
                f"{label}__{POSITIVE_SUFFIX}",
                "positive",
                replicate,
                n,
                requested,
                name,
                truth[name],
                result[name],
            )
            rows.append(positive)
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{label}__shrunken_se_control",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=positive["truth"],
                    estimate=positive["estimate"],
                    standard_error=SHRUNKEN_SE_FACTOR * positive["std_error"],
                    critical=CRITICAL,
                )
            )
        rows.extend(_joint_rows(law, result, truth, replicate=replicate, n=n, requested=requested))
        return rows

    if property_name == "mar_robustness":
        result = fit_cleverly(frame, law, **_robustness_learners(law, configuration))
        role = "control" if configuration == CONTROL else "positive"
        return [
            _estimand_row(
                property_name,
                f"{laws.label(law, name)}__{configuration}",
                role,
                replicate,
                n,
                requested,
                name,
                truth[name],
                result[name],
            )
            for name in laws.ESTIMANDS[law.key]
            if laws.stem(name) == "ate"
        ]

    if property_name == "crossfit_overfitting":
        label = cell.split("__", 1)[0]
        name = next(item for item in laws.ESTIMANDS[law.key] if laws.label(law, item) == label)
        result = fit_cleverly(
            _with_noise(frame, seed),
            law,
            outcome_learner=_fully_grown_outcome(law),
            # Both mechanisms read ``W`` from its own column and ignore the noise.
            treatment_learner=laws.LawTreatment(law),
            missingness_learner=laws.LawResponse(law),
            cross_fit=configuration != "in_sample",
            adjustment=("W", *NOISE_COLUMNS),
        )
        role = "control" if configuration == "in_sample" else "positive"
        return [
            _estimand_row(
                property_name, cell, role, replicate, n, requested, name, truth[name], result[name]
            )
        ]
    raise KeyError(property_name)  # pragma: no cover - declaration guard


def _specs() -> tuple[list[ReplicationSpec], list[list[ReplicationSpec]]]:
    """Return the independent specifications and the paired overfitting arms."""
    independent = [
        ReplicationSpec(
            "interval_calibration",
            f"{key}__calibration",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "learned_nuisances",
        )
        for key in laws.LAWS
    ]
    independent += [
        ReplicationSpec(
            "mar_robustness",
            f"{key}__{configuration}",
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            configuration,
        )
        for key in laws.LAWS
        for configuration in ROBUSTNESS_CONFIGURATIONS
    ]
    # Both arms of a pair deliberately share every draw; only the sample-splitting policy
    # changes.
    paired = [
        [
            ReplicationSpec(
                "crossfit_overfitting",
                f"{label}__{cell}",
                OVERFIT_N,
                OVERFIT_REPLICATES,
                configuration,
                seed_key=f"{label}__paired",
            )
            for cell, configuration in (
                ("stacked_arm_indexed_cvtmle", "cross_fit"),
                ("in_sample_control", "in_sample"),
            )
        ]
        for label in OVERFIT_LABELS
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
        paired = [[replace(spec, replicates=replicates) for spec in pair] for pair in paired]
    out = replication_payloads(STUDY, independent)
    for pair in paired:
        cross_fit, in_sample = (replication_payloads(STUDY, [spec]) for spec in pair)
        out += [payload for both in zip(cross_fit, in_sample, strict=True) for payload in both]
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    return pd.DataFrame([row for result in outcomes for row in result])


def generate_smoke_property_rows(*, n_jobs: int = 1) -> pd.DataFrame:
    """Fit one declared replication of every property cell without summarizing it."""
    outcomes = map_parallel(_fit_replication, _payloads(replicates=1), n_jobs=n_jobs)
    return pd.DataFrame([row for result in outcomes for row in result])


def simultaneous_coverage_verdicts(summary: pd.DataFrame, *, margins: Margins) -> None:
    """Read each joint-coverage cell against the rule its role answers to.

    The band must hold its exact joint-coverage interval inside the two-sided calibration
    coverage band, the rule every calibrated pointwise cell answers to.  The pointwise
    control must establish joint coverage below the nominal rate, which is what shows the
    band's wider critical value does work on these laws.
    """
    joint = summary["property"] == "simultaneous_coverage"
    for index in summary.index[joint]:
        coverage = Interval(
            float(summary.loc[index, "coverage_ci_lower"]),
            float(summary.loc[index, "coverage_ci_upper"]),
        )
        if summary.loc[index, "role"] == "control":
            passed = coverage.high < 1.0 - margins.alpha
        else:
            passed = coverage.within(*margins.calibration_coverage)
        summary.loc[index, "passed"] = bool(passed)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        rate_labels=(),
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE_SUFFIX)
    robustness_verdicts(summary, family="mar_robustness")
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    for label in OVERFIT_LABELS:
        crossfit_overfitting_verdicts(
            summary,
            rows,
            STUDY,
            positive_cell=f"{label}__stacked_arm_indexed_cvtmle",
            control_cell=f"{label}__in_sample_control",
        )
    return finish(summary, rates)


def declared_settings() -> dict[str, Any]:
    """Expose result-determining constants for the generated evidence document."""
    return {
        "robustness_replicates": ROBUSTNESS_REPLICATES,
        "robustness_n": ROBUSTNESS_N,
        "calibration_replicates": CALIBRATION_REPLICATES,
        "calibration_n": CALIBRATION_N,
        "overfit_replicates": OVERFIT_REPLICATES,
        "overfit_n": OVERFIT_N,
        "shrunken_se_factor": SHRUNKEN_SE_FACTOR,
        "nuisance_bound": NUISANCE_BOUND,
    }
