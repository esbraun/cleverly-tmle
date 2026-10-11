"""Recorded comparator failures, and the accounting that decides whether a cell survives them.

RM39 part a2.  An error that a comparator raises inside the runner's declared comparator-fit
stage (``study_stage`` in ``tests/canonical/study_harness.R``) costs one group, not the run.  The
harness writes a failure record in place of the group's checkpoint, and the driver publishes the
records as ``failures.csv``.  This module turns those records into a verdict.

A failed replicate is a missing value whose outcome nobody saw, so the accounting does not ask
whether the failures look independent of the result.  A test of that cannot see the missing
value.  It asks two questions of every cell that lost a replicate instead:

* did the cell lose more replicates than its declared bound allows
  (``StudyRecord.failure_bound``, ``floor(bound * replicates)``, 0 by default)?
* does every verdict leg still pass when the failed values take each value in a declared,
  unfavourable candidate set?

A cell fails when either answer is no.  The candidate sets are the rules of RM39-plan section
4.5 (B2, M1, m13) for the legs a reference failure can reach: the reference's own truth row and
the paired row.  Rates take every split of the failed replicates.  A bias or a paired difference
takes every split between the surviving minimum and maximum, with the margin's scale held at its
complete-case value.  A reference in a non-inferiority leg takes its best case.  The calibration
excess takes the attainable range of the reference's SE ratio.  Range-bounded imputation assumes
a failed value lies inside the surviving range; the bound limits that blind spot.

With no failure, nothing here runs and no artifact moves.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.inference import (
    bootstrap,
    clopper_pearson,
    lower_bound,
    percentile_interval,
    standardized_bias_verdict,
    student_interval,
    upper_bound,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import truth_on_inference_scale
from tests.studies.evidence.seeds import stream_seed

#: The published record file and the published accounting file.  Each exists only when a
#: failure exists, so a study without one publishes exactly the artifacts it did before.
FAILURES = "failures.csv"
ACCOUNTING = "failure-accounting.csv"

#: The columns of ``failures.csv``.  ``scenarios`` and ``replicates`` are ``;``-joined; a
#: scenario of ``*`` means the runner's frame carried no scenario column, so the group covered
#: every scenario the study declares.
FAILURE_COLUMNS = (
    "implementation",
    "group_id",
    "scenarios",
    "replicates",
    "attempt",
    "class",
    "stage",
    "call",
    "message",
    "seed",
)

#: The columns of ``failure-accounting.csv``: one row per cell and leg.  ``slack`` is the
#: distance by which a leg clears its margin, so a negative value fails.
ACCOUNTING_COLUMNS = (
    "table",
    "implementation",
    "scenario",
    "estimand",
    "requested",
    "failed",
    "rate",
    "bound",
    "allowed",
    "dropped_replications",
    "leg",
    "complete_case",
    "worst_case",
    "imputation",
    "leg_passed",
    "gate_passed",
    "reason",
    "witness_failed_error",
    "witness_succeeded_error",
)

#: The conclusion an otherwise passing paired row takes when its failures are not accounted for.
UNACCOUNTED = "unaccounted"

#: The reasons a cell's gate records.
WITHIN_BOUND = "within bound"
BOUND_EXCEEDED = "bound exceeded"
WORST_CASE_FAILS = "worst case fails"
NOT_ACCOUNTABLE = "failures not accountable"

#: Rounding slack on ``bound * replicates``: 0.01 * 300 is 2.9999999999999996 in floating point.
_FLOOR_SLACK = 1e-9


def allowed(record: StudyRecord, implementation: str) -> int:
    """How many replicates of one cell may fail before the cell is red."""
    bound = float(record.failure_bound.get(implementation, 0.0))
    return math.floor(bound * record.replicates + _FLOOR_SLACK)


def read(path: Path) -> pd.DataFrame | None:
    """The records in ``path``, or ``None`` when the file does not exist."""
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"scenarios": str, "replicates": str}, keep_default_na=False)
    return frame.loc[:, list(FAILURE_COLUMNS)]


def expand(failures: pd.DataFrame | None, record: StudyRecord) -> pd.DataFrame:
    """Every (implementation, scenario, estimand, replicate) a failure record covers.

    A record names the scenarios its group held, or ``*`` when the frame carried no scenario
    column.  It costs every estimand the study declares for each of those scenarios.
    """
    columns = ["implementation", "scenario", "estimand", "replicate"]
    if failures is None or failures.empty:
        return pd.DataFrame(columns=columns)
    out: list[tuple[str, str, str, int]] = []
    for row in failures.itertuples(index=False):
        scenarios = [part for part in str(row.scenarios).split(";") if part]
        if scenarios == ["*"]:
            scenarios = list(record.scenarios)
        unknown = sorted(set(scenarios) - set(record.scenarios))
        if unknown:
            raise ValueError(f"{record.slug}: failure {row.group_id} names unknown {unknown}")
        replicates = [int(part) for part in str(row.replicates).split(";") if part]
        for scenario in scenarios:
            for estimand in record.scenarios[scenario]:
                for replicate in replicates:
                    out.append((str(row.implementation), scenario, estimand, replicate))
    frame = pd.DataFrame(out, columns=columns)
    duplicated = frame.duplicated()
    if duplicated.any():
        raise ValueError(
            f"{record.slug}: two failure records cover one cell replicate:\n"
            f"{frame.loc[duplicated].to_string(index=False)}"
        )
    return frame


def reconcile(rows: pd.DataFrame, failures: pd.DataFrame | None, *, record: StudyRecord) -> None:
    """Refuse a replicate file whose missing rows are not exactly the recorded failures.

    Per cell, the surviving replicates and the failed ones are disjoint and together cover
    ``0 .. replicates - 1``.  A short cell without a record is refused, as it always was.  A cell
    with no survivor is refused, because nothing can be accounted for on an empty cell.
    """
    lost = expand(failures, record)
    subject = lost.loc[lost["implementation"] == record.implementation]
    if not subject.empty:
        raise ValueError(
            f"{record.slug}: {record.implementation} has recorded failures; Python-phase failures "
            f"arrive with RM39b"
        )
    unknown = set(lost["implementation"]) - set(record.implementations)
    if unknown:
        raise ValueError(f"{record.slug}: failures of undeclared implementations {sorted(unknown)}")
    every = set(range(record.replicates))
    failed = {
        key: set(group["replicate"].astype(int))
        for key, group in lost.groupby(["implementation", "scenario", "estimand"], sort=True)
    }
    wrong: list[str] = []
    for key, group in rows.groupby(["implementation", "scenario", "estimand"], sort=True):
        survived = list(group["replicate"].astype(int))
        lost_here = failed.pop(key, set())
        if len(survived) != len(set(survived)):
            wrong.append(f"{key}: a replicate appears twice")
            continue
        if set(survived) & lost_here:
            wrong.append(f"{key}: replicates both survived and failed")
        elif set(survived) | lost_here != every:
            wrong.append(
                f"{key}: {len(survived)} surviving and {len(lost_here)} failed replicates do not "
                f"cover the declared {record.replicates}"
            )
    for key in failed:
        wrong.append(f"{key}: every replicate failed, so nothing in this cell can be accounted for")
    if wrong:
        raise ValueError(
            f"{record.slug}: cells whose replicates are not the declared count:\n"
            + "\n".join(wrong)
        )


# ---- candidate imputations ------------------------------------------------------------------


def vertex(survivors: np.ndarray, k: int, j: int) -> np.ndarray:
    """``survivors`` with ``j`` failed values at their minimum and ``k - j`` at their maximum."""
    low, high = float(np.min(survivors)), float(np.max(survivors))
    return np.concatenate([survivors, np.full(j, low), np.full(k - j, high)])


def filled(survivors: np.ndarray, k: int, value: float) -> np.ndarray:
    """``survivors`` with all ``k`` failed values at ``value``."""
    return np.concatenate([survivors, np.full(k, float(value))])


@dataclass(frozen=True)
class Leg:
    """One verdict leg under its candidates: the complete-case and the least favourable value."""

    name: str
    complete_case: float
    worst_case: float
    imputation: str
    passed: bool


def _worst(name: str, complete_case: float, candidates: Iterable[tuple[str, float]]) -> Leg:
    """The candidate with the smallest slack; the leg passes when every slack is non-negative."""
    label, slack = min(candidates, key=lambda item: item[1])
    return Leg(name, complete_case, slack, label, bool(slack >= 0.0))


def _bias_slack(errors: np.ndarray, *, margin: float, confidence: float, scale: float) -> float:
    verdict = standardized_bias_verdict(
        errors, margin=margin, confidence_level=confidence, scale=scale
    )
    return float(verdict.margin - max(abs(verdict.interval.low), abs(verdict.interval.high)))


def _band_slack(interval: Any, low: float, high: float) -> float:
    return float(min(interval.low - low, high - interval.high))


def _se_ratio(
    spread: np.ndarray, std_errors: np.ndarray, *, replicates: int, seed: int
) -> np.ndarray:
    samples = bootstrap(
        {"spread": spread, "std_error": std_errors},
        {"ratio": lambda draw: draw["std_error"].mean(axis=1) / draw["spread"].std(axis=1, ddof=1)},
        replicates=replicates,
        seed=seed,
    )["ratio"]
    return samples


def truth_legs(group: pd.DataFrame, k: int, *, record: StudyRecord, seed: int) -> list[Leg]:
    """The reference's own truth row under its failures (m13: the subject rules, at extremes)."""
    margins = record.margins
    estimand = str(group["estimand"].iloc[0])
    inference = group["inference_estimate"].to_numpy(dtype=float)
    std_errors = group["std_error"].to_numpy(dtype=float)
    if record.truth_varies_by_replicate:
        errors = inference - group["truth"].to_numpy(dtype=float)
        spread = errors
    else:
        truth = float(group["truth"].iloc[0])
        errors = inference - truth_on_inference_scale(
            estimand, truth, str(group["inference_scale"].iloc[0])
        )
        spread = inference
    requested = len(group) + k
    confidence = margins.confidence_level

    complete = standardized_bias_verdict(
        errors, margin=margins.standardized_bias, confidence_level=confidence
    )
    bias = _worst(
        "bias",
        float(complete.margin - max(abs(complete.interval.low), abs(complete.interval.high))),
        (
            (
                f"j={j} at min",
                _bias_slack(
                    vertex(errors, k, j),
                    margin=margins.standardized_bias,
                    confidence=confidence,
                    scale=complete.scale,
                ),
            )
            for j in range(k + 1)
        ),
    )

    covered = int(group["covered"].sum())
    coverage = _worst(
        "coverage",
        float(
            clopper_pearson(covered, len(group), confidence_level=confidence).low
            - margins.coverage_floor
        ),
        (
            (
                f"j={j} covered",
                float(
                    clopper_pearson(covered + j, requested, confidence_level=confidence).low
                    - margins.coverage_floor
                ),
            )
            for j in range(k + 1)
        ),
    )

    low, high = margins.se_ratio_sanity
    replicates = margins.bootstrap_replicates

    def band(spread_values: np.ndarray, se_values: np.ndarray) -> float:
        samples = _se_ratio(spread_values, se_values, replicates=replicates, seed=seed)
        return _band_slack(percentile_interval(samples, confidence_level=confidence), low, high)

    mean_spread = float(np.mean(spread))
    candidates = [
        (
            "SE at max, estimates at mean",
            band(filled(spread, k, mean_spread), filled(std_errors, k, np.max(std_errors))),
        )
    ]
    candidates += [
        (
            f"SE at min, estimates j={j} at min",
            band(vertex(spread, k, j), filled(std_errors, k, np.min(std_errors))),
        )
        for j in range(k + 1)
    ]
    ratio = _worst("se ratio", band(spread, std_errors), candidates)
    return [bias, coverage, ratio]


def paired_legs(
    wide: pd.DataFrame,
    failed: Sequence[int],
    *,
    record: StudyRecord,
    truth: float,
    comparable: bool,
    seed: int,
) -> tuple[list[Leg], tuple[float, float]]:
    """The paired row's legs under the reference's failures, and the informative witness.

    ``wide`` holds every replicate of the subject (index = replicate) and the reference's
    surviving values, ``NaN`` where it failed.  The RMSE, coverage and calibration candidates
    keep the subject's real rows and impute only the reference; the similarity leg imputes the
    paired difference.
    """
    margins = record.margins
    subject, reference = record.implementation, str(record.reference)
    confidence = margins.confidence_level
    replicates = margins.bootstrap_replicates
    lost = wide.index.isin(list(failed))
    both = wide.loc[~lost]
    k = int(lost.sum())

    def column(frame: pd.DataFrame, name: str, implementation: str) -> np.ndarray:
        return frame[(name, implementation)].to_numpy(dtype=float)

    # Similarity: the difference at each vertex split, the pooled-SD margin held.
    difference = column(both, "estimate", subject) - column(both, "estimate", reference)
    pooled_sd = float(
        np.sqrt(
            0.5
            * (
                np.var(column(both, "estimate", subject), ddof=1)
                + np.var(column(both, "estimate", reference), ddof=1)
            )
        )
    )
    margin = margins.paired_difference * pooled_sd

    def similarity_slack(values: np.ndarray) -> float:
        interval = student_interval(values, confidence_level=confidence)
        return float(margin - max(abs(interval.low), abs(interval.high)))

    similarity = _worst(
        "similarity",
        similarity_slack(difference),
        ((f"j={j} at min", similarity_slack(vertex(difference, k, j))) for j in range(k + 1)),
    )

    # Subject real, reference ordered so that the failed replicates come last.
    order = np.concatenate([np.flatnonzero(~lost), np.flatnonzero(lost)])

    def subject_values(name: str) -> np.ndarray:
        return column(wide, name, subject)[order]

    def reference_survivors(name: str) -> np.ndarray:
        return column(both, name, reference)

    # RMSE ratio and coverage difference: the reference's best case.
    best = bootstrap(
        {
            "subject_estimate": subject_values("estimate"),
            "reference_estimate": filled(reference_survivors("estimate"), k, truth),
            "subject_covered": subject_values("covered"),
            "reference_covered": filled(reference_survivors("covered"), k, 1.0),
        },
        {
            "rmse_ratio": lambda draw: np.sqrt(
                np.mean((draw["subject_estimate"] - truth) ** 2, axis=1)
                / np.mean((draw["reference_estimate"] - truth) ** 2, axis=1)
            ),
            "coverage_difference": lambda draw: np.mean(
                draw["subject_covered"] - draw["reference_covered"], axis=1
            ),
        },
        replicates=replicates,
        seed=seed,
    )
    complete = bootstrap(
        {
            "subject_estimate": column(both, "estimate", subject),
            "reference_estimate": column(both, "estimate", reference),
            "subject_covered": column(both, "covered", subject),
            "reference_covered": column(both, "covered", reference),
        },
        {
            "rmse_ratio": lambda draw: np.sqrt(
                np.mean((draw["subject_estimate"] - truth) ** 2, axis=1)
                / np.mean((draw["reference_estimate"] - truth) ** 2, axis=1)
            ),
            "coverage_difference": lambda draw: np.mean(
                draw["subject_covered"] - draw["reference_covered"], axis=1
            ),
        },
        replicates=replicates,
        seed=seed,
    )
    rmse = _worst(
        "rmse ratio",
        float(
            margins.rmse_noninferiority
            - upper_bound(complete["rmse_ratio"], confidence_level=confidence)
        ),
        [
            (
                "reference estimate = truth",
                float(
                    margins.rmse_noninferiority
                    - upper_bound(best["rmse_ratio"], confidence_level=confidence)
                ),
            )
        ],
    )
    coverage = _worst(
        "coverage difference",
        float(
            lower_bound(complete["coverage_difference"], confidence_level=confidence)
            - margins.coverage_noninferiority
        ),
        [
            (
                "reference covered",
                float(
                    lower_bound(best["coverage_difference"], confidence_level=confidence)
                    - margins.coverage_noninferiority
                ),
            )
        ],
    )
    legs = [similarity, rmse, coverage]

    if comparable:
        subject_se = subject_values("std_error")
        subject_inference = subject_values("inference_estimate")
        survivors_se = reference_survivors("std_error")
        survivors_inference = reference_survivors("inference_estimate")

        estimate_sets = [
            ("estimates at mean", filled(survivors_inference, k, np.mean(survivors_inference)))
        ]
        estimate_sets += [
            (f"estimates j={j} at min", vertex(survivors_inference, k, j)) for j in range(k + 1)
        ]
        low_se, high_se = float(np.min(survivors_se)), float(np.max(survivors_se))
        candidates: list[tuple[str, float]] = []
        for label, inference_values in estimate_sets:
            spread = float(np.std(inference_values, ddof=1))
            # The common failed SE that puts the reference's full-sample ratio at 1, or the
            # closest attainable value inside the surviving range.
            root = (spread * (len(survivors_se) + k) - float(np.sum(survivors_se))) / k
            for se_label, value in (
                ("SE at min", low_se),
                ("SE at max", high_se),
                ("SE at root", min(max(root, low_se), high_se)),
            ):
                candidates.append(
                    (
                        f"{se_label}, {label}",
                        _calibration_slack(
                            subject_se,
                            subject_inference,
                            filled(survivors_se, k, value),
                            inference_values,
                            record=record,
                            seed=seed,
                        ),
                    )
                )
        both_se = column(both, "std_error", reference)
        both_inference = column(both, "inference_estimate", reference)
        complete_calibration = _calibration_slack(
            column(both, "std_error", subject),
            column(both, "inference_estimate", subject),
            both_se,
            both_inference,
            record=record,
            seed=seed,
        )
        legs.append(_worst("calibration excess", complete_calibration, candidates))

    errors = column(wide, "estimate", subject) - truth
    witness = (
        float(np.mean(errors[lost])) if k else math.nan,
        float(np.mean(errors[~lost])),
    )
    return legs, witness


def _calibration_slack(
    subject_se: np.ndarray,
    subject_inference: np.ndarray,
    reference_se: np.ndarray,
    reference_inference: np.ndarray,
    *,
    record: StudyRecord,
    seed: int,
) -> float:
    margins = record.margins

    def excess(draw: dict[str, np.ndarray]) -> np.ndarray:
        subject_ratio = draw["subject_se"].mean(axis=1) / draw["subject_inference"].std(
            axis=1, ddof=1
        )
        reference_ratio = draw["reference_se"].mean(axis=1) / draw["reference_inference"].std(
            axis=1, ddof=1
        )
        return np.abs(subject_ratio - 1.0) - np.abs(reference_ratio - 1.0)

    samples = bootstrap(
        {
            "subject_se": subject_se,
            "subject_inference": subject_inference,
            "reference_se": reference_se,
            "reference_inference": reference_inference,
        },
        {"excess": excess},
        replicates=margins.bootstrap_replicates,
        seed=seed,
    )
    return float(
        margins.calibration_noninferiority
        - upper_bound(samples["excess"], confidence_level=margins.confidence_level)
    )


# ---- the accounting -------------------------------------------------------------------------


def _gate(failed: int, allowance: int, legs: Sequence[Leg]) -> tuple[bool, str]:
    if failed > allowance:
        return False, BOUND_EXCEEDED
    if not all(leg.passed for leg in legs):
        return False, WORST_CASE_FAILS
    return True, WITHIN_BOUND


def _rows(
    *,
    table: str,
    implementation: str,
    scenario: str,
    estimand: str,
    record: StudyRecord,
    failed: int,
    dropped: int,
    legs: Sequence[Leg],
    witness: tuple[float, float] = (math.nan, math.nan),
) -> list[dict[str, Any]]:
    allowance = allowed(record, implementation)
    passed, reason = _gate(failed, allowance, legs)
    bound = Leg("failure bound", float(failed), float(allowance), "", failed <= allowance)
    return [
        {
            "table": table,
            "implementation": implementation,
            "scenario": scenario,
            "estimand": estimand,
            "requested": record.replicates,
            "failed": failed,
            "rate": failed / record.replicates,
            "bound": float(record.failure_bound.get(implementation, 0.0)),
            "allowed": allowance,
            "dropped_replications": dropped,
            "leg": leg.name,
            "complete_case": leg.complete_case,
            "worst_case": leg.worst_case,
            "imputation": leg.imputation,
            "leg_passed": leg.passed,
            "gate_passed": passed,
            "reason": reason,
            "witness_failed_error": witness[0],
            "witness_succeeded_error": witness[1],
        }
        for leg in (bound, *legs)
    ]


def _truth_payload(
    payload: tuple[StudyRecord, pd.DataFrame, tuple[str, str, str], int, int],
) -> list[dict[str, Any]]:
    record, group, key, k, seed = payload
    implementation, scenario, estimand = key
    legs = truth_legs(group, k, record=record, seed=seed)
    return _rows(
        table="performance",
        implementation=implementation,
        scenario=scenario,
        estimand=estimand,
        record=record,
        failed=k,
        dropped=k,
        legs=legs,
    )


def _paired_payload(
    payload: tuple[StudyRecord, pd.DataFrame, tuple[str, str], tuple[int, ...], int],
) -> list[dict[str, Any]]:
    record, group, key, failed, seed = payload
    scenario, estimand = key
    reference = str(record.reference)
    wide = group.pivot(
        index="replicate",
        columns="implementation",
        values=["estimate", "covered", "inference_estimate", "std_error"],
    ).sort_index()
    legs, witness = paired_legs(
        wide,
        failed,
        record=record,
        truth=float(group["truth"].iloc[0]),
        comparable=estimand not in record.incomparable_se,
        seed=seed,
    )
    return _rows(
        table="equivalence",
        implementation=reference,
        scenario=scenario,
        estimand=estimand,
        record=record,
        failed=len(failed),
        dropped=len(failed),
        legs=legs,
        witness=witness,
    )


def truth_accounting(
    rows: pd.DataFrame,
    failures: pd.DataFrame | None,
    *,
    record: StudyRecord,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """The accounting rows of every truth cell that lost a replicate."""
    lost = expand(failures, record)
    if lost.empty:
        return pd.DataFrame(columns=list(ACCOUNTING_COLUMNS))
    counts = lost.groupby(["implementation", "scenario", "estimand"], sort=True).size()
    payloads = []
    for key, k in counts.items():
        implementation, scenario, estimand = (str(part) for part in key)
        group = rows.loc[
            (rows["implementation"] == implementation)
            & (rows["scenario"] == scenario)
            & (rows["estimand"] == estimand)
        ].sort_values("replicate")
        seed = stream_seed(record, "failure-accounting", "performance", *key)
        payloads.append(((record, group, (implementation, scenario, estimand), int(k), seed),))
    out = [row for rows_ in map_parallel(_truth_payload, payloads, n_jobs=n_jobs) for row in rows_]
    return pd.DataFrame.from_records(out, columns=list(ACCOUNTING_COLUMNS))


def paired_accounting(
    rows: pd.DataFrame,
    failures: pd.DataFrame | None,
    *,
    record: StudyRecord,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """The accounting rows of every paired cell whose reference lost a replicate."""
    lost = expand(failures, record)
    if lost.empty or record.reference is None:
        return pd.DataFrame(columns=list(ACCOUNTING_COLUMNS))
    lost = lost.loc[lost["implementation"] == record.reference]
    payloads = []
    for key, group in lost.groupby(["scenario", "estimand"], sort=True):
        scenario, estimand = str(key[0]), str(key[1])
        cell = rows.loc[(rows["scenario"] == scenario) & (rows["estimand"] == estimand)]
        failed = tuple(sorted(int(value) for value in group["replicate"]))
        seed = stream_seed(record, "failure-accounting", "equivalence", scenario, estimand)
        payloads.append(((record, cell, (scenario, estimand), failed, seed),))
    out = [row for rows_ in map_parallel(_paired_payload, payloads, n_jobs=n_jobs) for row in rows_]
    return pd.DataFrame.from_records(out, columns=list(ACCOUNTING_COLUMNS))


def hook_accounting(
    study: Any,
    failures: pd.DataFrame | None,
    extra_frames: Mapping[str, pd.DataFrame],
    *,
    record: StudyRecord,
) -> pd.DataFrame:
    """One row per hook audit of a study with failures (M3).

    A study module may define ``account_failures(failures, extra_frames)``, returning one row per
    audit with the columns ``leg``, ``failed`` (rows the audit counts as failing, a failed
    group's missing rows included) and ``passed`` (its own threshold applied).  A study with a
    hook and no such function gets one ``failures not accountable`` row, which is red.
    """
    columns = list(ACCOUNTING_COLUMNS)
    if failures is None or failures.empty:
        return pd.DataFrame(columns=columns)
    hooks = [
        name
        for name in ("scientific_failures", "extra_artifacts", "reference_artifacts")
        if hasattr(study, name)
    ]
    if not hooks:
        return pd.DataFrame(columns=columns)
    implementation = str(record.reference)
    failed = len(expand(failures, record)[["scenario", "replicate"]].drop_duplicates())
    base = {
        "table": "hook",
        "implementation": implementation,
        "scenario": "",
        "estimand": "",
        "requested": record.replicates,
        "failed": failed,
        "rate": math.nan,
        "bound": float(record.failure_bound.get(implementation, 0.0)),
        "allowed": allowed(record, implementation),
        "dropped_replications": failed,
        "complete_case": math.nan,
        "worst_case": math.nan,
        "imputation": "",
        "witness_failed_error": math.nan,
        "witness_succeeded_error": math.nan,
    }
    if not hasattr(study, "account_failures"):
        return pd.DataFrame.from_records(
            [
                {
                    **base,
                    "leg": "/".join(hooks),
                    "leg_passed": False,
                    "gate_passed": False,
                    "reason": NOT_ACCOUNTABLE,
                }
            ],
            columns=columns,
        )
    audits = study.account_failures(failures, extra_frames)
    out = []
    for audit in audits.itertuples(index=False):
        passed = bool(audit.passed)
        out.append(
            {
                **base,
                "leg": str(audit.leg),
                "complete_case": float(audit.failed),
                "leg_passed": passed,
                "gate_passed": passed,
                "reason": WITHIN_BOUND if passed else WORST_CASE_FAILS,
            }
        )
    return pd.DataFrame.from_records(out, columns=columns)


def cell_gates(accounting: pd.DataFrame, table: str) -> dict[tuple[str, ...], tuple[bool, str]]:
    """Each accounted cell's gate and reason, keyed as its verdict table keys it."""
    frame = accounting.loc[accounting["table"] == table]
    keys = (
        ["implementation", "scenario", "estimand"]
        if table == "performance"
        else ["scenario", "estimand"]
    )
    out: dict[tuple[str, ...], tuple[bool, str]] = {}
    for key, group in frame.groupby(keys, sort=True):
        out[tuple(str(part) for part in key)] = (
            bool(group["gate_passed"].iloc[0]),
            str(group["reason"].iloc[0]),
        )
    return out


def correct_performance(performance: pd.DataFrame, accounting: pd.DataFrame) -> pd.DataFrame:
    """``performance-tests.csv`` with each accounting-red cell's ``passed`` set false."""
    gates = cell_gates(accounting, "performance")
    if not gates:
        return performance
    out = performance.copy()
    for index, row in out.iterrows():
        key = (str(row["implementation"]), str(row["scenario"]), str(row["estimand"]))
        if key in gates and not gates[key][0]:
            out.at[index, "passed"] = False
    return out


def correct_paired(paired: pd.DataFrame, accounting: pd.DataFrame) -> pd.DataFrame:
    """``equivalence.csv`` with each accounting-red cell's verdict and conclusion corrected."""
    gates = cell_gates(accounting, "equivalence")
    if not gates:
        return paired
    out = paired.copy()
    for index, row in out.iterrows():
        key = (str(row["scenario"]), str(row["estimand"]))
        if key in gates and not gates[key][0]:
            if bool(row["passed"]):
                out.at[index, "comparison_conclusion"] = UNACCOUNTED
            out.at[index, "passed"] = False
    return out


def red(accounting: pd.DataFrame) -> pd.DataFrame:
    """The accounting rows of every cell whose gate failed, one per cell and leg."""
    return accounting.loc[~accounting["gate_passed"].astype(bool)]
