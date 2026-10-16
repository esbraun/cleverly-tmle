"""Recorded comparator failures, and the accounting that decides whether a cell survives them.

RM39 parts a2 and b.  An error that a comparator raises inside the runner's declared
comparator-fit stage (``study_stage`` in ``tests/canonical/study_harness.R``) costs one group, not
the run.  A recorded ``cleverly`` fit failure in the Python phase costs one replicate
(``tests/studies/evidence/replication.py``).  The driver publishes both sides' records as
``failures.csv``.  This module turns those records into a verdict.

A failed replicate is a missing value whose outcome nobody saw, so the accounting does not ask
whether the failures look independent of the result.  A test of that cannot see the missing
value.  It asks two questions of every cell that lost a replicate instead:

* did the cell lose more replicates than its declared bound allows
  (``StudyRecord.failure_bound``, ``floor(bound * replicates)``, 0 by default)?
* does every verdict leg still pass when the failed values take each value in a declared,
  unfavourable candidate set?

A cell fails when either answer is no.  The candidate sets are the rules of RM39-plan section
4.5 (B2, M1, m13) and RM39b-plan sections 7 and 14: each implementation's own truth row and the paired
row.  Rates take every split of the failed replicates.  A bias or a paired difference takes every
split between the surviving minimum and maximum, with the margin's scale held at its
complete-case value.  In a non-inferiority leg the reference takes its best case and the subject
its worst.  The calibration excess takes the product of the two sides' attainable SE-ratio
ranges.  Range-bounded imputation assumes a failed value lies inside the surviving range; the
bound limits that blind spot.  A reference record of class ``paired_input_missing`` costs the
reference a replicate whose input the subject's failed fit never built, so it counts against the
subject's bound.

With no failure, nothing here runs and no artifact moves.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
    "subject_failed",
    "reference_failed",
    "witness_reference_error_on_subject_failed",
    "witness_reference_error_on_shared",
)

#: The class of a reference record the driver writes for a replicate the reference never saw,
#: because its input is built from the subject's fit and that fit failed (RM39b B1).  It costs
#: the reference's cells the replicate, but it counts against the subject's bound.
PAIRED_INPUT_MISSING = "paired_input_missing"

#: The conclusion an otherwise passing paired row takes when its failures are not accounted for.
UNACCOUNTED = "unaccounted"

#: The reasons a cell's gate records.
WITHIN_BOUND = "within bound"
BOUND_EXCEEDED = "bound exceeded"
WORST_CASE_FAILS = "worst case fails"
NOT_ACCOUNTABLE = "failures not accountable"
#: Every leg that fails its worst case fails its complete case too: the verdict is red on its
#: own, and the failures do not decide it.
COMPLETE_CASE_FAILS = "complete case fails"
#: A paired cell whose equivalence route survives and whose superiority route does not.
SUPERIORITY_LOST = "superiority not accounted"
#: A paired cell whose superiority route survives and whose equivalence route does not.
EQUIVALENCE_LOST = "equivalence not accounted"

#: The paired legs each passing route needs (``comparison.comparison_verdict``).  The
#: calibration leg exists only where the standard errors are comparable.
EQUIVALENCE_ROUTE = ("similarity", "rmse ratio", "coverage difference", "calibration excess")
SUPERIORITY_ROUTE = ("coverage superiority", "rmse ratio", "calibration excess")

#: Rounding slack on ``bound * replicates``: 0.01 * 300 is 2.9999999999999996 in floating point.
_FLOOR_SLACK = 1e-9


def allowance(bound: float, replicates: int) -> int:
    """How many of ``replicates`` may fail under ``bound``: ``floor(bound * replicates)``."""
    return math.floor(bound * replicates + _FLOOR_SLACK)


def allowed(record: StudyRecord, implementation: str) -> int:
    """How many replicates of one cell may fail before the cell is red."""
    return allowance(float(record.failure_bound.get(implementation, 0.0)), record.replicates)


#: The manifest configuration key that records the primary bounds a run applied.
BOUND_KEY = "failure_bound"


def applied_bounds(manifest: Path) -> dict[str, float]:
    """The primary bounds a committed run applied, from its manifest.

    A run writes ``configuration.failure_bound`` when its record declares a bound.  A manifest
    without the key is a run under bound 0 for every implementation.

    Parameters
    ----------
    manifest : pathlib.Path
        The study's ``manifest.json``.

    Returns
    -------
    dict
        Implementation to bound; empty when the file or the key is absent.
    """
    if not manifest.exists():
        return {}
    configuration = json.loads(manifest.read_text()).get("configuration") or {}
    return {str(name): float(bound) for name, bound in (configuration.get(BOUND_KEY) or {}).items()}


def bound_consistent(
    configuration: Mapping[str, Any], record: StudyRecord, *, failures_recorded: bool
) -> bool:
    """Whether a committed manifest's primary bounds agree with the record (RM39b2).

    A stated key must equal the record's ``failure_bound``.  A manifest without the key is a
    run under bound 0.  It is consistent only when the run recorded no failure, because then
    no bound decided any of its verdicts.  A record may declare a bound for its next run.

    Parameters
    ----------
    configuration : Mapping
        The manifest's ``configuration``.
    record : StudyRecord
        The study.
    failures_recorded : bool
        Whether the study commits ``failures.csv``.

    Returns
    -------
    bool
        ``True`` when the manifest and the record agree.
    """
    if BOUND_KEY in configuration:
        return dict(configuration[BOUND_KEY]) == dict(record.failure_bound)
    return not failures_recorded


def raised_after_failures(here: Path, record: StudyRecord) -> list[str]:
    """The implementations whose bound a run would raise after a committed run lost replicates.

    A committed ``failures.csv`` shows which sides lost a replicate.  A re-run draws the same
    seeds and meets the same failures, so a bound above the one the committed run applied
    (:func:`applied_bounds`) would buy the pass.  A ``paired_input_missing`` record counts
    against the subject, as the accounting counts it.

    Parameters
    ----------
    here : pathlib.Path
        The study's artifact directory.
    record : StudyRecord
        The study as the run declares it.

    Returns
    -------
    list of str
        The implementations whose declared bound exceeds the applied one, sorted.
    """
    applied = applied_bounds(here / "manifest.json")
    raised = {
        side
        for side, bound in record.failure_bound.items()
        if float(bound) > applied.get(side, 0.0)
    }
    if not raised:
        return []
    recorded = read(here / FAILURES)
    if recorded is None or recorded.empty:
        return []
    sides = set(recorded["implementation"].astype(str))
    if (recorded["class"].astype(str) == PAIRED_INPUT_MISSING).any():
        sides.add(record.implementation)
    return sorted(sides & raised)


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


def induced(subject: pd.DataFrame, *, record: StudyRecord) -> pd.DataFrame:
    """One ``paired_input_missing`` reference record per subject record.

    A study whose reference input comes from the subject's fit writes no sample for a
    replicate the subject lost, so the reference loses it too.  Each record names the
    subject's group in its message and keeps its group id, scenarios, replicates and seed.
    """
    if record.reference is None:
        raise ValueError(f"{record.slug}: no reference to lose a paired input")
    out = subject.loc[:, list(FAILURE_COLUMNS)].copy()
    out["implementation"] = record.reference
    out["attempt"] = 1
    out["class"] = PAIRED_INPUT_MISSING
    out["stage"] = "subject fit"
    out["call"] = ""
    out["message"] = [
        f"the subject's fit {group} failed, so this replicate has no reference input"
        for group in subject["group_id"]
    ]
    return out.reset_index(drop=True)


def reconcile(rows: pd.DataFrame, failures: pd.DataFrame | None, *, record: StudyRecord) -> None:
    """Refuse a replicate file whose missing rows are not exactly the recorded failures.

    Per cell, the surviving replicates and the failed ones are disjoint and together cover
    ``0 .. replicates - 1``.  A short cell without a record is refused, as it always was.  A cell
    with no survivor is refused, because nothing can be accounted for on an empty cell.
    """
    lost = expand(failures, record)
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
    """One implementation's own truth row under its lost replicates (m13: the rules, at extremes)."""
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
    subject_failed: Sequence[int] = (),
) -> tuple[list[Leg], tuple[float, float, float, float]]:
    """The paired row's legs under both sides' failures, and the informative witnesses.

    ``wide`` holds one row per replicate (index = replicate) and each side's surviving values,
    ``NaN`` where that side failed.  ``failed`` lists the reference's failed replicates and
    ``subject_failed`` the subject's; a replicate may be in both.  The similarity leg imputes
    the paired difference over their union.  The RMSE, coverage and calibration candidates keep
    each side's real values where it survived and fill the rest: the reference at its best
    case, the subject at its worst.  The calibration excess takes the product of the two
    sides' candidate sets.  With no subject failure, every candidate is the RM39a2 one.

    The witnesses are the subject's mean error on the replicates only the reference lost and on
    the replicates both kept, then the reference's on the replicates only the subject lost and
    on those both kept.
    """
    margins = record.margins
    subject, reference = record.implementation, str(record.reference)
    confidence = margins.confidence_level
    replicates = margins.bootstrap_replicates
    reference_lost = wide.index.isin(list(failed))
    subject_lost = wide.index.isin(list(subject_failed))
    lost = reference_lost | subject_lost
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

    # Every replicate, with the ones either side lost last, so the rows stay paired by position.
    order = np.concatenate([np.flatnonzero(~lost), np.flatnonzero(lost)])
    masks = {subject: subject_lost[order], reference: reference_lost[order]}
    k_side = {name: int(mask.sum()) for name, mask in masks.items()}

    def full(name: str, implementation: str) -> np.ndarray:
        return column(wide, name, implementation)[order]

    def survivors(name: str, implementation: str) -> np.ndarray:
        return full(name, implementation)[~masks[implementation]]

    def fill(name: str, implementation: str, values: Any) -> np.ndarray:
        out = full(name, implementation).copy()
        out[masks[implementation]] = values
        return out

    # The subject's worst RMSE: every failed estimate at the surviving one farthest from the
    # truth.  The ratio rises in each subject squared error, so no range-bounded fill is worse.
    subject_estimates = survivors("estimate", subject)
    farthest = float(subject_estimates[np.argmax(np.abs(subject_estimates - truth))])
    rmse_parts, coverage_parts = [], []
    if k_side[subject]:
        rmse_parts.append("subject estimate at farthest")
        coverage_parts.append("subject missed")
    if k_side[reference]:
        rmse_parts.append("reference estimate = truth")
        coverage_parts.append("reference covered")
    rmse_label, coverage_label = "; ".join(rmse_parts), "; ".join(coverage_parts)
    statistics = {
        "rmse_ratio": lambda draw: np.sqrt(
            np.mean((draw["subject_estimate"] - truth) ** 2, axis=1)
            / np.mean((draw["reference_estimate"] - truth) ** 2, axis=1)
        ),
        "coverage_difference": lambda draw: np.mean(
            draw["subject_covered"] - draw["reference_covered"], axis=1
        ),
    }
    worst = bootstrap(
        {
            "subject_estimate": fill("estimate", subject, farthest),
            "reference_estimate": fill("estimate", reference, truth),
            "subject_covered": fill("covered", subject, 0.0),
            "reference_covered": fill("covered", reference, 1.0),
        },
        statistics,
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
        statistics,
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
                rmse_label,
                float(
                    margins.rmse_noninferiority
                    - upper_bound(worst["rmse_ratio"], confidence_level=confidence)
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
                coverage_label,
                float(
                    lower_bound(worst["coverage_difference"], confidence_level=confidence)
                    - margins.coverage_noninferiority
                ),
            )
        ],
    )
    # Superiority needs a lower bound strictly above zero, so its leg passes on a positive slack.
    superiority_complete = float(
        lower_bound(complete["coverage_difference"], confidence_level=confidence)
    )
    superiority_worst = float(
        lower_bound(worst["coverage_difference"], confidence_level=confidence)
    )
    superiority = Leg(
        "coverage superiority",
        superiority_complete,
        superiority_worst,
        coverage_label,
        superiority_worst > 0.0,
    )
    legs = [similarity, rmse, coverage, superiority]

    if comparable:
        legs.append(
            _worst(
                "calibration excess",
                _calibration_slack(
                    column(both, "std_error", subject),
                    column(both, "inference_estimate", subject),
                    column(both, "std_error", reference),
                    column(both, "inference_estimate", reference),
                    record=record,
                    seed=seed,
                ),
                [
                    (
                        "; ".join(part for part in (subject_label, reference_label) if part),
                        _calibration_slack(
                            subject_se,
                            subject_inference,
                            reference_se,
                            reference_inference,
                            record=record,
                            seed=seed,
                        ),
                    )
                    for subject_label, subject_se, subject_inference in _subject_calibration(
                        survivors, fill, subject, k_side[subject]
                    )
                    for reference_label, reference_se, reference_inference in (
                        _reference_calibration(survivors, fill, reference, k_side[reference])
                    )
                ],
            )
        )

    subject_errors = full("estimate", subject) - truth
    reference_errors = full("estimate", reference) - truth
    only_reference = masks[reference] & ~masks[subject]
    only_subject = masks[subject] & ~masks[reference]
    kept = ~(masks[subject] | masks[reference])

    def mean_of(values: np.ndarray, where: np.ndarray) -> float:
        return float(np.mean(values[where])) if where.any() else math.nan

    witness = (
        mean_of(subject_errors, only_reference),
        mean_of(subject_errors, kept),
        mean_of(reference_errors, only_subject),
        mean_of(reference_errors, kept),
    )
    return legs, witness


_Fill = Callable[[str, str, Any], np.ndarray]


def _subject_calibration(
    survivors: Callable[[str, str], np.ndarray], fill: _Fill, subject: str, k: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The subject's (label, SEs, inference estimates) candidates for the calibration excess.

    The subject's excess grows as its SE ratio leaves 1 in either direction, so the candidates
    are the two edges of the RM39a2 truth-row SE-ratio rule: the failed SEs at the survivor
    maximum with the estimates at the survivor mean, and at the survivor minimum with the
    estimates at each vertex split.
    """
    if not k:
        return [("", fill("std_error", subject, []), fill("inference_estimate", subject, []))]
    se = survivors("std_error", subject)
    inference = survivors("inference_estimate", subject)
    out = [
        (
            "subject SE at max, estimates at mean",
            fill("std_error", subject, float(np.max(se))),
            fill("inference_estimate", subject, float(np.mean(inference))),
        )
    ]
    out += [
        (
            f"subject SE at min, estimates j={j} at min",
            fill("std_error", subject, float(np.min(se))),
            fill(
                "inference_estimate",
                subject,
                np.concatenate([np.full(j, np.min(inference)), np.full(k - j, np.max(inference))]),
            ),
        )
        for j in range(k + 1)
    ]
    return out


def _reference_calibration(
    survivors: Callable[[str, str], np.ndarray], fill: _Fill, reference: str, k: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The reference's candidates for the calibration excess (RM39a2 B2).

    Its inference estimates at the survivor mean or at each vertex split, crossed with a
    common failed SE at the survivor minimum, the maximum, or the root that puts its
    full-sample SE ratio at 1, clipped to the surviving range.
    """
    if not k:
        return [("", fill("std_error", reference, []), fill("inference_estimate", reference, []))]
    se = survivors("std_error", reference)
    inference = survivors("inference_estimate", reference)
    estimate_sets = [
        ("estimates at mean", fill("inference_estimate", reference, float(np.mean(inference))))
    ]
    estimate_sets += [
        (
            f"estimates j={j} at min",
            fill(
                "inference_estimate",
                reference,
                np.concatenate([np.full(j, np.min(inference)), np.full(k - j, np.max(inference))]),
            ),
        )
        for j in range(k + 1)
    ]
    low_se, high_se = float(np.min(se)), float(np.max(se))
    out = []
    for label, inference_values in estimate_sets:
        spread = float(np.std(inference_values, ddof=1))
        # The common failed SE that puts the reference's full-sample ratio at 1, or the
        # closest attainable value inside the surviving range.
        root = (spread * (len(se) + k) - float(np.sum(se))) / k
        for se_label, value in (
            ("SE at min", low_se),
            ("SE at max", high_se),
            ("SE at root", min(max(root, low_se), high_se)),
        ):
            out.append(
                (f"{se_label}, {label}", fill("std_error", reference, value), inference_values)
            )
    return out


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


def _complete(leg: Leg) -> bool:
    """Whether the leg passes on its complete cases; superiority needs a positive bound."""
    if leg.name == "coverage superiority":
        return leg.complete_case > 0.0
    return leg.complete_case >= 0.0


def _gate(failed: int, allowance: int, legs: Sequence[Leg]) -> tuple[bool, str]:
    """A truth cell: red when the failures exceed the bound or a leg fails its worst case.

    The reason separates a leg the failures turned red from one red on its complete cases
    (plan 4.4), so the ledger does not attribute a complete-case failure to the failures.
    """
    if failed > allowance:
        return False, BOUND_EXCEEDED
    failing = [leg for leg in legs if not leg.passed]
    if not failing:
        return True, WITHIN_BOUND
    if any(_complete(leg) for leg in failing):
        return False, WORST_CASE_FAILS
    return False, COMPLETE_CASE_FAILS


def _paired_gate(failed: int, allowance: int, legs: Sequence[Leg]) -> tuple[bool, str]:
    """A paired cell: green when some route that passes on the complete cases survives.

    A route survives when every one of its legs passes its worst case.  The reason names a
    route that held on the complete cases and did not survive, so that
    :func:`accounted_conclusion` can withdraw that route's conclusion alone.
    """
    if failed > allowance:
        return False, BOUND_EXCEEDED
    by_name = {leg.name: leg for leg in legs}
    routes = {
        name: [by_name[leg] for leg in route if leg in by_name]
        for name, route in (("equivalent", EQUIVALENCE_ROUTE), ("superior", SUPERIORITY_ROUTE))
    }
    survives = {name: all(leg.passed for leg in route) for name, route in routes.items()}
    held = {name: all(_complete(leg) for leg in route) for name, route in routes.items()}
    lost = {name for name in routes if held[name] and not survives[name]}
    if any(survives.values()):
        if not lost:
            return True, WITHIN_BOUND
        return True, SUPERIORITY_LOST if "superior" in lost else EQUIVALENCE_LOST
    if any(held.values()):
        return False, WORST_CASE_FAILS
    return False, COMPLETE_CASE_FAILS


def accounted_conclusion(
    conclusion: str, *, equivalent: bool, gate: tuple[bool, str]
) -> tuple[str, bool]:
    """A paired cell's conclusion and verdict once its failure accounting applies.

    Parameters
    ----------
    conclusion : str
        The conclusion the cell's own legs give, before the accounting.
    equivalent : bool
        Whether the cell's own legs also give the equivalence conclusion.
    gate : tuple of (bool, str)
        The cell's accounting gate and reason, as :func:`cell_gates` returns them.

    Returns
    -------
    tuple of (str, bool)
        The published conclusion and whether the cell passes.
    """
    if conclusion not in ("superior", "equivalent"):
        return conclusion, False
    passed, reason = gate
    if not passed:
        return UNACCOUNTED, False
    if conclusion == "superior" and reason == SUPERIORITY_LOST:
        return ("equivalent", True) if equivalent else (UNACCOUNTED, False)
    if conclusion == "equivalent" and reason == EQUIVALENCE_LOST:
        return UNACCOUNTED, False
    return conclusion, True


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
    counted: Mapping[str, int],
    witness: tuple[float, ...] = (math.nan, math.nan, math.nan, math.nan),
    gate: Any = None,
) -> list[dict[str, Any]]:
    """One row per leg of a cell, after one ``failure bound`` leg per side that failed.

    ``counted`` maps each implementation to the failures its own bound counts.  The reference's
    bound leg is always written, as in RM39a2; the subject's only when the subject failed.  The
    cell's gate is red when any side exceeds its bound.
    """
    bounds = []
    # The row's own implementation first, so its leg reads as the RM39a2 bound.
    for side, count in sorted(counted.items(), key=lambda item: item[0] != implementation):
        if side != implementation and not count:
            continue
        allowance = allowed(record, side)
        name = "failure bound" if side == implementation else f"{_side(record, side)} failure bound"
        bounds.append(Leg(name, float(count), float(allowance), "", count <= allowance))
    over = int(not all(leg.passed for leg in bounds))
    passed, reason = (gate or _gate)(over, 0, legs)
    subject = int(counted.get(record.implementation, 0))
    reference = int(counted.get(str(record.reference), 0)) if record.reference else 0
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
            "allowed": allowed(record, implementation),
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
            "subject_failed": subject,
            "reference_failed": reference,
            "witness_reference_error_on_subject_failed": witness[2],
            "witness_reference_error_on_shared": witness[3],
        }
        for leg in (*bounds, *legs)
    ]


def _side(record: StudyRecord, implementation: str) -> str:
    return "subject" if implementation == record.implementation else "reference"


def _own(failures: pd.DataFrame | None, record: StudyRecord) -> pd.DataFrame:
    """The expansion of the records each implementation's own bound counts.

    A ``paired_input_missing`` record costs the reference a replicate the subject's failure
    took, so it counts against the subject: it is relabelled to the subject here.
    """
    if failures is None or failures.empty:
        return expand(failures, record)
    relabelled = failures.copy()
    induced = relabelled["class"] == PAIRED_INPUT_MISSING
    relabelled.loc[induced, "implementation"] = record.implementation
    lost = expand(relabelled.loc[~induced], record)
    induced_lost = expand(relabelled.loc[induced], record)
    return pd.concat([lost, induced_lost], ignore_index=True).drop_duplicates(ignore_index=True)


def _truth_payload(
    payload: tuple[StudyRecord, pd.DataFrame, tuple[str, str, str], int, int, int],
) -> list[dict[str, Any]]:
    record, group, key, k, own, seed = payload
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
        counted={implementation: own},
    )


def _paired_payload(
    payload: tuple[
        StudyRecord,
        pd.DataFrame,
        tuple[str, str],
        tuple[int, ...],
        tuple[int, ...],
        Mapping[str, int],
        int,
    ],
) -> list[dict[str, Any]]:
    record, group, key, reference_failed, subject_failed, counted, seed = payload
    scenario, estimand = key
    reference = str(record.reference)
    wide = group.pivot(
        index="replicate",
        columns="implementation",
        values=["estimate", "covered", "inference_estimate", "std_error"],
    ).sort_index()
    replicates = pd.Index(range(record.replicates), name="replicate")
    wide = wide.reindex(replicates)
    legs, witness = paired_legs(
        wide,
        reference_failed,
        record=record,
        truth=float(group["truth"].iloc[0]),
        comparable=estimand not in record.incomparable_se,
        seed=seed,
        subject_failed=subject_failed,
    )
    union = len(set(reference_failed) | set(subject_failed))
    return _rows(
        table="equivalence",
        implementation=reference,
        scenario=scenario,
        estimand=estimand,
        record=record,
        failed=union,
        dropped=union,
        legs=legs,
        counted=counted,
        witness=witness,
        gate=_paired_gate,
    )


def truth_accounting(
    rows: pd.DataFrame,
    failures: pd.DataFrame | None,
    *,
    record: StudyRecord,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """The accounting rows of every truth cell that lost a replicate.

    The legs take every replicate the cell lost.  The bound counts the cell's own failures:
    a reference cell's ``paired_input_missing`` replicates count against the subject.
    """
    lost = expand(failures, record)
    if lost.empty:
        return pd.DataFrame(columns=list(ACCOUNTING_COLUMNS))
    keys = ["implementation", "scenario", "estimand"]
    counts = lost.groupby(keys, sort=True).size()
    own = _own(failures, record).groupby(keys, sort=True).size()
    payloads = []
    for key, k in counts.items():
        implementation, scenario, estimand = (str(part) for part in key)
        group = rows.loc[
            (rows["implementation"] == implementation)
            & (rows["scenario"] == scenario)
            & (rows["estimand"] == estimand)
        ].sort_values("replicate")
        seed = stream_seed(record, "failure-accounting", "performance", *key)
        counted = int(own.get(key, 0))
        payloads.append(
            ((record, group, (implementation, scenario, estimand), int(k), counted, seed),)
        )
    out = [row for rows_ in map_parallel(_truth_payload, payloads, n_jobs=n_jobs) for row in rows_]
    return pd.DataFrame.from_records(out, columns=list(ACCOUNTING_COLUMNS))


def _replicates(
    frame: pd.DataFrame, implementation: str, scenario: str, estimand: str
) -> tuple[int, ...]:
    chosen = frame.loc[
        (frame["implementation"] == implementation)
        & (frame["scenario"] == scenario)
        & (frame["estimand"] == estimand)
    ]
    return tuple(sorted(int(value) for value in chosen["replicate"]))


def paired_accounting(
    rows: pd.DataFrame,
    failures: pd.DataFrame | None,
    *,
    record: StudyRecord,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """The accounting rows of every paired cell where either side lost a replicate."""
    lost = expand(failures, record)
    # A descriptive paired row states no verdict, so a lost replicate has none to correct.
    if lost.empty or record.reference is None or record.paired_role == "descriptive":
        return pd.DataFrame(columns=list(ACCOUNTING_COLUMNS))
    subject, reference = record.implementation, str(record.reference)
    own = _own(failures, record)
    payloads = []
    for key, group in lost.groupby(["scenario", "estimand"], sort=True):
        scenario, estimand = str(key[0]), str(key[1])

        cell = rows.loc[(rows["scenario"] == scenario) & (rows["estimand"] == estimand)]
        counted = {
            subject: len(_replicates(own, subject, scenario, estimand)),
            reference: len(_replicates(own, reference, scenario, estimand)),
        }
        seed = stream_seed(record, "failure-accounting", "equivalence", scenario, estimand)
        payloads.append(
            (
                (
                    record,
                    cell,
                    (scenario, estimand),
                    _replicates(group, reference, scenario, estimand),
                    _replicates(group, subject, scenario, estimand),
                    counted,
                    seed,
                ),
            )
        )
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
    group's missing rows included), ``passed`` (its own threshold applied) and, optionally,
    ``own`` (the surviving rows that fail it).  A study with a
    hook and no such function gets one ``failures not accountable`` row, which is red.

    ``implementation`` names the sides whose own records the rows count, joined by ``+``, and
    ``subject_failed`` and ``reference_failed`` count each side's lost replications.  A
    ``paired_input_missing`` record is the subject's.  ``bound`` and ``allowed`` are the named
    side's, or ``NaN`` when both sides failed, because each side has its own bound.
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
    own = _own(failures, record)
    counts = {
        side: len(
            own.loc[own["implementation"] == side, ["scenario", "replicate"]].drop_duplicates()
        )
        for side in record.implementations
    }
    sides = [side for side in record.implementations if counts[side]]
    if not sides:
        raise ValueError(f"{record.slug}: failure records that name no declared implementation")
    failed = missing_fits(failures, record)
    (single,) = sides if len(sides) == 1 else (None,)
    base = {
        "table": "hook",
        "implementation": "+".join(sides),
        # A hook audit has no scenario, estimand or imputation.  The CSV writes NaN as an empty
        # cell and reads an empty column back as NaN, so the in-memory frame matches the
        # committed one.
        "scenario": math.nan,
        "estimand": math.nan,
        "requested": record.replicates,
        "failed": failed,
        "rate": math.nan,
        "bound": math.nan if single is None else float(record.failure_bound.get(single, 0.0)),
        "allowed": math.nan if single is None else allowed(record, single),
        "dropped_replications": failed,
        "complete_case": math.nan,
        "worst_case": math.nan,
        "imputation": math.nan,
        "witness_failed_error": math.nan,
        "witness_succeeded_error": math.nan,
        "subject_failed": counts[record.implementation],
        "reference_failed": counts[record.reference] if record.reference is not None else 0,
        "witness_reference_error_on_subject_failed": math.nan,
        "witness_reference_error_on_shared": math.nan,
    }
    if not hasattr(study, "account_failures"):
        return pd.DataFrame.from_records(
            [
                {
                    **base,
                    "leg": "+".join(hooks),
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
        own = int(getattr(audit, "own", 0))
        reason = WITHIN_BOUND if passed else COMPLETE_CASE_FAILS if own else WORST_CASE_FAILS
        out.append(
            {
                **base,
                "leg": str(audit.leg),
                "complete_case": float(audit.failed),
                "leg_passed": passed,
                "gate_passed": passed,
                "reason": reason,
            }
        )
    return pd.DataFrame.from_records(out, columns=columns)


def missing_fits(
    failures: pd.DataFrame | None, record: StudyRecord, implementation: str | None = None
) -> int:
    """The (scenario, replicate) fits the failure records left without a row.

    With ``implementation``, the fits that implementation lost, a ``paired_input_missing``
    record included.  Without it, a replication that either side lost counts once.
    """
    lost = expand(failures, record)
    if implementation is not None:
        lost = lost.loc[lost["implementation"] == implementation]
    return len(lost[["scenario", "replicate"]].drop_duplicates())


def audits(gates: Mapping[str, pd.DataFrame], *, missing: Mapping[str, int]) -> pd.DataFrame:
    """The ``account_failures`` frame of a study whose audits fail on any row.

    ``gates`` is the study's own ``scientific_failures`` output.  ``missing`` names the audits
    a failed fit would have entered, with the count of fits it lost: each one counts as a
    failing row of that audit (decision Q1).  An audit passes when nothing fails it.
    ``own`` is the count of surviving rows that fail it, which the ledger reads to tell an
    audit the failures turned red from one red on its own rows.
    """
    unknown = sorted(set(missing) - set(gates))
    if unknown:
        raise ValueError(f"no audit named {unknown}")
    out = []
    for leg, frame in gates.items():
        failed = len(frame) + int(missing.get(leg, 0))
        out.append({"leg": leg, "failed": failed, "passed": failed == 0, "own": len(frame)})
    return pd.DataFrame.from_records(out, columns=["leg", "failed", "passed", "own"])


@contextlib.contextmanager
def surviving_samples(samples: Path, reference_results: Path) -> Iterator[Path]:
    """The samples of the replications the reference returned, for a probe to read.

    A probe that reruns the comparator's arithmetic on every reference replication skips the
    replications whose fit failed (decision Q1).  A replication is skipped when its scenario
    has reference rows and that replication has none; the validation of the reference rows
    has already shown that the returned and failed replications together cover the run.

    Without a skipped replication this yields ``samples`` itself, so a run with no failure
    feeds the probe the bytes it always read.  Otherwise it writes the kept lines beside
    ``samples``, because the container mounts that one directory, and deletes them on exit.
    The lines are copied as text: a parse and a rewrite would move the last bits of a float.
    """
    returned = pd.read_csv(reference_results, usecols=["scenario", "replicate"])
    pairs = {(str(scenario), int(replicate)) for scenario, replicate in returned.itertuples(False)}
    scenarios = {scenario for scenario, _ in pairs}
    with gzip.open(samples, "rt", encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    at_scenario, at_replicate = header.index("scenario"), header.index("replicate")
    kept = [lines[0]]
    for line, fields in zip(lines[1:], csv.reader(lines[1:]), strict=True):
        scenario, replicate = fields[at_scenario], int(float(fields[at_replicate]))
        if scenario in scenarios and (scenario, replicate) not in pairs:
            continue
        kept.append(line)
    if len(kept) == len(lines):
        yield samples
        return
    path = samples.with_name(f"surviving-{samples.name}")
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write("".join(kept).encode("utf-8"))
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def probe_audits(
    probe: pd.DataFrame,
    failures: pd.DataFrame | None,
    *,
    record: StudyRecord,
    scenarios: Iterable[str],
    coverage: str,
    audit: str,
) -> pd.DataFrame:
    """The ``account_failures`` frame of a probe that skips the failed replications.

    The coverage audit passes when every probe path holds one row for each declared
    replication ``0, ..., R - 1`` of ``scenarios`` that the reference did not lose, and the
    lost ones fill the rest.  The probe reruns the reference's arithmetic, so a replication
    the subject alone lost stays in it.
    The ``audit`` leg counts the probe rows whose ``passed`` is not true.  A failed
    replication has no comparison row, so no published number depends on its scale
    workaround.
    """
    scenarios = set(scenarios)
    lost = expand(failures, record)
    failed = {
        int(replicate)
        for scenario, replicate in lost.loc[lost["implementation"] == record.reference][
            ["scenario", "replicate"]
        ]
        .drop_duplicates()
        .itertuples(index=False)
        if scenario in scenarios
    }
    paths = probe.groupby("path") if "path" in probe.columns else [("", probe)]
    gaps = 0
    for _, rows in paths:
        probed = [int(value) for value in rows["replicate"]]
        if len(set(probed)) != len(probed) or set(probed) & failed:
            gaps += record.replicates
        else:
            gaps += len(set(range(record.replicates)) ^ (set(probed) | failed))
    if probe.empty or not set(probe["scenario"]) <= scenarios:
        gaps = max(gaps, 1)
    failing = int((~probe["passed"].fillna(False).astype(bool)).sum()) if len(probe) else 0
    out = [
        {"leg": coverage, "failed": gaps, "passed": gaps == 0},
        {"leg": audit, "failed": failing, "passed": failing == 0},
    ]
    return pd.DataFrame.from_records(out, columns=["leg", "failed", "passed"])


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
    """``equivalence.csv`` with each accounted cell's verdict and conclusion corrected."""
    gates = cell_gates(accounting, "equivalence")
    if not gates:
        return paired
    out = paired.copy()
    for index, row in out.iterrows():
        key = (str(row["scenario"]), str(row["estimand"]))
        if key not in gates:
            continue
        conclusion, passed = accounted_conclusion(
            str(row["comparison_conclusion"]),
            equivalent=bool(row["paired_similarity"]) and bool(row["subject_not_inferior"]),
            gate=gates[key],
        )
        out.at[index, "comparison_conclusion"] = conclusion
        out.at[index, "passed"] = passed
    return out


def red(accounting: pd.DataFrame) -> pd.DataFrame:
    """The accounting rows of every cell whose gate failed, one per cell and leg."""
    return accounting.loc[~accounting["gate_passed"].astype(bool)]
