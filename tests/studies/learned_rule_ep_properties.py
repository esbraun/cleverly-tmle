"""Repeated-sampling properties of the learned-rule value with the EP rule learner (P1 5.5).

The gated family, ``interval_calibration``, fits at n = 4,000 (:data:`CALIBRATION_N`), twice the
primary size, where the learned-rule-learners pilot found the rule-learner SE deficit within
about 3% of zero (``tests/canonical/learned_rule_learners/README.md``).  Two families report
and state no verdict: ``small_sample_diagnostic`` at n = 1,000 and ``exceptional_law_report``
at n = 2,000.

``<law>__declared`` for ``non_exceptional`` and ``misspecified_limit``
    The declared fit.  Its SE-ratio interval must lie inside 0.93 to 1.07 and its coverage
    interval inside 0.92 to 0.98, both read on the error ``estimate - truth``, where the truth
    is the fold-average value of the rules the fit learned (rule L4).
``optimal_known_g__declared``
    The value of the optimal rule, ``ey_optimal_rule[learned rule]``, with the EP rule given
    both adjustment columns (R12 admits it) and ``g0`` supplied, at ``non_exceptional``.  Its
    truth is the fixed :math:`\\Psi_{d_0}(P_0)`.  The EP limit rule is the sign of the additive
    projection of the effect, which differs from ``d0`` on 1.0% of the population near the
    threshold, and its regret is about 2.5e-5, 0.002 SE at n = 4,000
    (:func:`~tests.studies._learned_rule_ep_law.optimal_regret`), so condition C3f holds up to
    a bias no budget here resolves.  The same band as the declared cells.
``<label>__shrunken_se_control`` and ``<label>__noise_control``
    The two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls` for each of the three
    labels: the SE shrunken by 0.70, and noise of one efficiency-bound unit added to each
    estimate.  Each must fall below the SE-ratio band.  They check the instrument.
``non_exceptional__validation_rule_control``
    The fold-locality control: :class:`~tests.studies._learned_rule_law.ValidationRowRule`
    fits fold ``v``'s EP curve on fold ``v``'s validation rows from the outer cross-fitted
    nuisances, at n = 2,000 (:data:`CONTROL_N`) for :data:`CONTROL_REPLICATES` draws of its own.
    Its truth is the value of the rules it learned.  Its coverage interval must lie wholly below
    the band floor 0.92.  It shows that the instrument sees a gross fold-locality violation;
    the subtle leaks (outer nuisances at ``T_v``, a size step that sees its inner fold) rest
    on the unit witnesses W-c2 and W-c3.

``small_sample_diagnostic/<law>__n1000``
    The declared fit and the paired plug-in fit at n = 1,000, 2,000 draws per law, role
    ``diagnostic``: the summary publishes the SE ratio with its bootstrap interval, the
    coverage and the bias with their intervals, and no margin is read.
``exceptional_law_report/exceptional__n2000``
    The declared fit at the RM30 ``exceptional`` law, which has no effect anywhere, so C3
    fails: the fold curves have no fixed sign.  2,400 draws, role ``diagnostic``.  The declared
    reading (:func:`exceptional_reading`) is the boundary study's: an upper end of the 99%
    Clopper-Pearson coverage interval below 0.95 reads "under-covers at the exceptional law", a
    lower end at or above 0.90 reads "no under-coverage resolved at the declared budget", and
    anything else reads "unresolved".  At 2,400 draws the first two meet at 2,252 covered, so
    the cell always resolves one way or the other.  The plug-in rule's boundary study measured coverage
    0.894 there; this cell reports the EP rule's.

Each row of the gated laws and of the diagnostic cells also carries the descriptive
``plug_in_estimate``, ``plug_in_std_error`` and ``plug_in_truth`` of the plug-in fit on the same
sample and fold seed; the optimal and control rows carry none.  The summary publishes
``value_gain_mean``, ``value_gain_sd`` and ``value_gain_positive_share`` on each gated law's
positive cell, and no verdict reads them.

A property draw's sample seed is ``stream_seed(record, "property_sample", family, label,
replicate, *suffix)`` and its fold seed the same tuple with ``"fold_partition"`` first; rule L3
of the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_ep_law as law
from tests.studies import _learned_rule_law as base
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import Interval, clopper_pearson
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    payload_cell,
    payload_replicate,
    ratio_intervals,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_ep import (
    CALIBRATION_LABELS,
    CONTROL_KIND,
    CONTROL_LABEL,
    DIAGNOSTIC_FAMILY,
    DIAGNOSTIC_KIND,
    EFFICIENCY_BOUNDS,
    EXCEPTIONAL_CELL,
    EXCEPTIONAL_FAMILY,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STUDY,
)

CALIBRATION_N = 4_000


def sizing_reading(ratio: float, sigma: float) -> tuple[int, bool]:
    """The plan 5.5 rule: ``R`` per calibration cell from the pilot ratio minus one pilot SE.

    At least 0.975 gives 4,800; 0.96 to 0.975 gives 7,200; below 0.96 gives 7,200 still, and
    the ``T49`` rows stay experimental whatever the run shows ("the calibration cells are not
    powered at the measured SE deficit").  Returns ``R`` and whether the rows may be added.
    """
    low = ratio - sigma
    if low >= 0.975:
        return 4_800, True
    if low >= 0.96:
        return 7_200, True
    return 7_200, False


#: Set from the ``non_exceptional`` sizing pilot by :func:`sizing_reading`
#: (``tests/unit/test_learned_rule_ep_design.py`` recomputes both from the committed pilot).
#: The pilot ratio is 0.970 over 2,000 draws, one pilot SE is 0.017, and 0.953 lies below 0.96:
#: R = 7,200, and the ``T49`` rows stay experimental whatever the run shows, because the
#: calibration cells are not powered at the measured SE deficit (plan 5.5).
CALIBRATION_REPLICATES = 7_200
T49_ADMITTED = False
CONTROL_N = PRIMARY_N
CONTROL_REPLICATES = 500
DIAGNOSTIC_N = 1_000
DIAGNOSTIC_REPLICATES = 2_000
EXCEPTIONAL_N = PRIMARY_N
EXCEPTIONAL_REPLICATES = 2_400
SHRUNKEN_SE_FACTOR = 0.70
POSITIVE = "declared"
OPTIMAL_LABEL = "optimal_known_g"

#: The descriptive summary columns of the paired comparison with the plug-in rule.
VALUE_GAIN_SUMMARY = ("value_gain_mean", "value_gain_sd", "value_gain_positive_share")

#: The reporting families, whose rows carry the role ``diagnostic``.
REPORTING_FAMILIES = (DIAGNOSTIC_FAMILY, EXCEPTIONAL_FAMILY)

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    *(
        base.PropertyDraw(
            "interval_calibration", f"{name}__{POSITIVE}", CALIBRATION_REPLICATES, name
        )
        for name in law.LAWS
    ),
    base.PropertyDraw(
        "interval_calibration",
        f"{OPTIMAL_LABEL}__{POSITIVE}",
        CALIBRATION_REPLICATES,
        "non_exceptional",
    ),
    base.PropertyDraw(
        "interval_calibration",
        f"{CONTROL_LABEL}__{CONTROL_KIND}",
        CONTROL_REPLICATES,
        CONTROL_LABEL,
    ),
    *(
        base.PropertyDraw(
            DIAGNOSTIC_FAMILY, f"{name}__{DIAGNOSTIC_KIND}", DIAGNOSTIC_REPLICATES, name
        )
        for name in law.LAWS
    ),
    base.PropertyDraw(
        EXCEPTIONAL_FAMILY, EXCEPTIONAL_CELL, EXCEPTIONAL_REPLICATES, law.EXCEPTIONAL_LAW
    ),
)


def size_of(family: str, label: str) -> int:
    """The sample size of a property cell's draws."""
    if family == DIAGNOSTIC_FAMILY:
        return DIAGNOSTIC_N
    if family == EXCEPTIONAL_FAMILY:
        return EXCEPTIONAL_N
    if label.endswith(f"__{CONTROL_KIND}"):
        return CONTROL_N
    return CALIBRATION_N


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*base.HARNESS_COLUMNS, *law.VALUE_GAIN_COLUMNS)

_NO_GAIN = dict.fromkeys(law.VALUE_GAIN_COLUMNS, float("nan"))

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, name = key[0]
    return base.draw(name, n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    family, label = key[0][:2]
    name = f"{family}/{label}"
    return [name, *DERIVED_CELLS.get(name, ())]


def _row(
    family: str,
    cell: str,
    role: str,
    key: tuple[int, int, int],
    truth: float,
    estimate: Any,
    harness: tuple[float, int, int],
    gain: dict[str, float],
) -> dict[str, Any]:
    replicate, n, requested = key
    row = replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    oracle, checked, warnings = harness
    row.update(
        {"oracle_se": oracle, "rule_rows_checked": checked, "solver_warnings": warnings, **gain}
    )
    return row


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, n, requested, _, fold_seed, name = key[0]
    where = (replicate, n, requested)
    if label == f"{OPTIMAL_LABEL}__{POSITIVE}":
        optimal = law.optimal_measured(frame, fold_seed)
        refit = optimal.refit
        return [
            _row(
                family,
                label,
                "positive",
                where,
                optimal.truth,
                optimal.result.estimates[law.OPTIMAL_ESTIMAND],
                (refit.oracle_se, refit.rows_checked, optimal.warnings),
                _NO_GAIN,
            )
        ]
    if label.endswith(f"__{CONTROL_KIND}"):
        result, truth, warnings = law.control(frame, name, fold_seed)
        return [
            _row(
                family,
                label,
                "control",
                where,
                truth.truth,
                result.estimates[law.ESTIMAND],
                (truth.oracle_se, truth.rows_checked, warnings),
                _NO_GAIN,
            )
        ]
    fitted = law.measured(frame, name, fold_seed)
    if family in REPORTING_FAMILIES:
        role = DIAGNOSTIC_ROLE
    elif family == "interval_calibration":
        role = "positive"
    else:
        raise ValueError(f"unknown property family {family!r}")
    truth = fitted.truth
    return [
        _row(
            family,
            label,
            role,
            where,
            truth.truth,
            fitted.result.estimates[law.ESTIMAND],
            (truth.oracle_se, truth.rows_checked, fitted.warnings),
            law.value_gain_columns(fitted),
        )
    ]


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            sample_seed = stream_seed(
                record, "property_sample", draw.family, draw.label, replicate, *suffix
            )
            fold_seed = stream_seed(
                record, "fold_partition", draw.family, draw.label, replicate, *suffix
            )
            out.append(
                (
                    (
                        draw.family,
                        draw.label,
                        replicate,
                        size_of(draw.family, draw.label),
                        replicates,
                        sample_seed,
                        fold_seed,
                        draw.law,
                    ),
                )
            )
    return out


FAILURE_BOUND = 0.01
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY, cap: int | None = None
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication."""
    outcomes = fit_properties(
        _payloads(record, cap),
        _replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    controls = calibration_controls(
        rows,
        record,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_BOUNDS,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=base.CRITICAL,
        positive_suffix=POSITIVE,
    )
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


def value_gain(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Fill the descriptive :data:`VALUE_GAIN_SUMMARY` of each gated law's positive cell."""
    for name in law.LAWS:
        cell = f"{name}__{POSITIVE}"
        group = rows.loc[(rows["property"] == "interval_calibration") & (rows["cell"] == cell)]
        gain = group["truth"].to_numpy(float) - group["plug_in_truth"].to_numpy(float)
        mask = (summary["property"] == "interval_calibration") & (summary["cell"] == cell)
        summary.loc[mask, "value_gain_mean"] = float(np.mean(gain))
        summary.loc[mask, "value_gain_sd"] = float(np.std(gain, ddof=1))
        summary.loc[mask, "value_gain_positive_share"] = float(np.mean(gain > 0.0))


def report_diagnostics(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Report the SE ratio of each reporting cell with its interval; state no verdict.

    The summary already holds the coverage and bias intervals of every cell.  This adds the
    bootstrap interval of the SE ratio, read on the error ``estimate - truth``, and writes
    ``passed`` and ``property_passed`` ``True`` on every reporting row, meaning the numbers
    were computed and are published; the published role says that no margin was read.
    """
    margins = STUDY.margins
    for family_name in REPORTING_FAMILIES:
        family = summary["property"] == family_name
        roles = set(summary.loc[family, "role"].astype(str))
        if family.any() and roles != {DIAGNOSTIC_ROLE}:
            raise ValueError(f"{family_name} publishes roles {sorted(roles)}, not diagnostic")
        for cell in sorted(summary.loc[family, "cell"]):
            group = rows.loc[(rows["property"] == family_name) & (rows["cell"] == cell)]
            ratio = ratio_intervals(
                group,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, family_name, cell),
                truth_varies=STUDY.truth_varies_by_replicate,
            )["se_ratio"]
            mask = family & (summary["cell"] == cell)
            summary.loc[mask, "se_ratio_ci_lower"] = ratio.low
            summary.loc[mask, "se_ratio_ci_upper"] = ratio.high
            summary.loc[mask, "passed"] = True
            summary.loc[mask, "property_passed"] = True


#: The readings of the exceptional cell, declared before any draw (plan 5.5, as the boundary
#: study's).
UNDER_COVERS = "under-covers at the exceptional law"
NO_UNDER_COVERAGE = "no under-coverage resolved at the declared budget"
UNRESOLVED = "unresolved"


def exceptional_reading(covered: int, replicates: int) -> tuple[str, Interval]:
    """The declared reading of the exceptional cell from its covered count.

    The 99% Clopper-Pearson coverage interval is read from the top: an upper end below 0.95
    reads :data:`UNDER_COVERS`, a lower end at or above 0.90 reads :data:`NO_UNDER_COVERAGE`,
    and anything else reads :data:`UNRESOLVED`.
    """
    interval = clopper_pearson(covered, replicates, confidence_level=STUDY.margins.confidence_level)
    if interval.high < 0.95:
        return UNDER_COVERS, interval
    if interval.low >= 0.90:
        return NO_UNDER_COVERAGE, interval
    return UNRESOLVED, interval


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, every statistic on the error of each row.

    The study has one calibration size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=VALUE_GAIN_SUMMARY, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    value_gain(summary, rows)
    report_diagnostics(summary, rows)
    return finish(summary, rates)


#: The declared rules of the failure accounting.  The reporting families write their verdict
#: columns without reading a margin, so only the calibration family has a rule.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)
