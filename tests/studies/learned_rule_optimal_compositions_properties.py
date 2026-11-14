"""Repeated-sampling properties of the compositions of the value of the optimal rule.

Two families, in the order the declaration reads them. Every fit uses the known treatment
mechanism and RM30's settings, and every cell reads the error against its label's fixed truth.

``interval_calibration``
    One draw label per setting (``weighted``, ``strata``, ``known_regimes``, ``in_sample``,
    ``intermediate``); each draw writes the positive cell of each of its labels
    (``<label>__declared``, nine in all), and each positive gets the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    :data:`SHRUNKEN_SE_FACTOR`, and noise of one efficiency-bound unit added to each estimate.
    Each stratum label also carries ``<label>__unnormalised_curve_control``, the same fit read
    with the SE of its stratum curve without the fold share ``1 / P_{n,v}(W2 = s)``, about half
    the reported SE, which must fall below the band.
``weighted_target_control``
    Its own draws of the ``weighted`` setting, each read twice: against the unweighted
    :math:`\\Psi_{d_0}(P_0)` (``target_gap_control``, 8.36 SEs below the target) and against
    the weighted value of the unweighted rule :math:`\\Psi^w_{d_0}(P_w)`
    (``unweighted_rule_control``, 2.05 SEs below). Each exact coverage interval must lie wholly
    below the calibration floor.

Every cell's summary also publishes the mean regret, the mean, smallest and largest
``rule_disagreement`` over its draws, and the standardized bias, coverage and SE ratio against
the value of the fit's own fold rules (:data:`OPTIMAL_SUMMARY`), read by no verdict. On the
``in_sample`` cell that standardized bias is the in-sample lean of the plan (bias against
:math:`\\Psi_{d_n}`). A property draw's sample seed is
``stream_seed(record, "property_sample", family, label, replicate, *suffix)`` and its fold seed
the same tuple with ``"fold_partition"`` first; rule L3 of the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _fold_evaluated_strata_law as fold_strata
from tests.studies import _learned_rule_optimal_compositions_law as law
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    cell_groups,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_optimal_compositions import (
    CALIBRATION_LABELS,
    EFFICIENCY_BOUNDS,
    POSITIVE,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STRATUM_LABELS,
    STUDY,
    TARGET_FAMILY,
    TARGET_LABEL,
    TARGET_TRUTHS,
    UNNORMALISED,
)

base = law.base

CALIBRATION_N = PRIMARY_N
#: The draws of each setting's calibration label, each draw serving all of the setting's labels.
#: A positive passes when both its SE-ratio leg and its coverage leg pass, so the count is sized
#: on the joint family (design review M2): the arm-pooled SE-ratio family of
#: :func:`tests.studies.mtp_study_tools.pass_probabilities` (pooled mean and ``tau`` from
#: :func:`~tests.studies.mtp_study_tools.replicates_from_pilot`) times the product of the nine
#: coverage legs, each predictive over its pilot coverage, read on the pilot of
#: :mod:`tests.studies.learned_rule_optimal_compositions_pilot`. On the 2,000-draw pilot every
#: label passes the per-arm screen, the pooled SE ratio is 1.004 (labels 0.980 to 1.028) and
#: ``tau`` 0.014. The SE-ratio rule alone gives 8,400 (0.951); the joint family first reaches
#: 0.95 at 9,600 (0.952), which is marginal under a stress like T46's (the pooled mean one pooled
#: pilot SE further from 1 and every coverage one pilot SE toward its nearer edge, read as known:
#: 0.949), so the count is the next multiple of 400, 10,000 (joint 0.957, stressed 0.954).
#: ``test_learned_rule_optimal_compositions_design`` recomputes all of it. The pooling is
#: optimistic where labels' SE constructions differ: ``strata_s1`` (pilot 1.028) passes with
#: about 0.79 at 10,000 if its true ratio is one pilot SD higher (the README's pooling caveat).
CALIBRATION_REPLICATES = 10_000
#: The ``weighted_target_control`` draws: the study floor of 1,000 as a multiple of 400; the
#: design test computes the control's power at this count from the quadrature gap, the tilted
#: curve's SD and n, and requires at least 0.90 (plan section 5).
TARGET_REPLICATES = 1_200
#: The plan's shrunken-SE control: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80

#: The draw label of the target control.
TARGET_DRAW = "weighted_control"

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    *(
        base.PropertyDraw("interval_calibration", setting, CALIBRATION_REPLICATES, setting)
        for setting in law.SETTINGS
    ),
    base.PropertyDraw(TARGET_FAMILY, TARGET_DRAW, TARGET_REPLICATES, "weighted"),
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arm to the controls derived from its rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = law.HARNESS_COLUMNS

#: The integer harness columns, which a derived control must keep integer.
INTEGER_COLUMNS = ("rule_rows_checked", "solver_warnings")

#: The descriptive summary columns of every cell (as in ``learned-rule-optimal-value``).
OPTIMAL_SUMMARY = (
    "mean_regret",
    "mean_rule_disagreement",
    "min_rule_disagreement",
    "max_rule_disagreement",
    "rule_value_standardized_bias",
    "rule_value_coverage",
    "rule_value_se_ratio",
)

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, setting = key[0]
    return law.draw(setting, n, sample_seed)


def target_truth(kind: str) -> float:
    """The truth a target control reads: ``unweighted`` or ``unweighted_rule``."""
    return {
        "unweighted": law.unweighted_truth,
        "unweighted_rule": law.unweighted_rule_truth,
    }[kind]()


def unnormalised_se(fitted: law.Measured, label: str) -> float:
    """The SE of the stratum curve without its ``1 / P_{n,v}(W2 = s)`` normalisation.

    The stored curve of the stratum parameter times each row's fold share, read with the
    fold-evaluated variance rule (:func:`tests.studies._fold_evaluated_strata_law.curve_se`),
    as ``learned-rule-strata`` builds the control.
    """
    result = fitted.results[label]
    curve = np.asarray(fitted.estimate(label).influence_curve, dtype=float)
    code = int(label[-1])
    return fold_strata.curve_se(curve * fold_strata.fold_shares(result, code))


def draw_cells(family: str, label: str) -> list[str]:
    """Every ``property/cell`` one draw writes, its derived controls included."""
    if family == TARGET_FAMILY:
        return [f"{family}/{cell}" for cell in PROPERTY_CELLS[family]]
    written = []
    for name in law.PARAMETERS[label]:
        positive = f"interval_calibration/{name}__{POSITIVE}"
        written += [positive, *DERIVED_CELLS[positive]]
        if name in STRATUM_LABELS:
            written.append(f"interval_calibration/{name}__{UNNORMALISED}")
    return written


def _cells_of(key: _Key) -> list[str]:
    return draw_cells(key[0][0], key[0][1])


def draw_rows(
    fitted: law.Measured, family: str, where: tuple[int, int, int]
) -> list[dict[str, Any]]:
    """The rows one draw writes: a positive row per label and a control row per stratum label,
    or the target controls' two rows."""
    replicate, n, requested = where
    rows = []
    if family == TARGET_FAMILY:
        estimate = fitted.estimate(TARGET_LABEL)
        for control, truth in TARGET_TRUTHS.items():
            row = control_row(
                property_name=family,
                cell=f"{TARGET_LABEL}__{control}",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=target_truth(truth),
                estimate=float(estimate.psi),
                standard_error=float(estimate.std_error),
                critical=base.CRITICAL,
                role="control",
            )
            row.update(law.harness_columns(fitted, TARGET_LABEL))
            rows.append(row)
        return rows
    for label in law.PARAMETERS[fitted.setting]:
        row = replicate_row(
            property_name=family,
            cell=f"{label}__{POSITIVE}",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=law.truth(label),
            estimate=fitted.estimate(label),
            alpha=STUDY.margins.alpha,
        )
        row.update(law.harness_columns(fitted, label))
        rows.append(row)
        if label in STRATUM_LABELS:
            unnormalised = control_row(
                property_name=family,
                cell=f"{label}__{UNNORMALISED}",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=law.truth(label),
                estimate=float(fitted.estimate(label).psi),
                standard_error=unnormalised_se(fitted, label),
                critical=base.CRITICAL,
            )
            unnormalised.update(law.harness_columns(fitted, label))
            rows.append(unnormalised)
    return rows


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, _, replicate, n, requested, _, fold_seed, setting = key[0]
    if family not in PROPERTY_CELLS:
        raise ValueError(f"unknown property family {family!r}")
    if n != CALIBRATION_N or setting not in law.SETTINGS:
        raise ValueError(f"{family} draws a declared setting at n = {CALIBRATION_N}")
    if family == TARGET_FAMILY and setting != "weighted":
        raise ValueError(f"{TARGET_FAMILY} draws the weighted setting, not {setting}")
    fitted = law.measured(frame, setting, fold_seed)
    return draw_rows(fitted, family, (replicate, n, requested))


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            sample_seed, fold_seed = (
                stream_seed(record, stream, draw.family, draw.label, replicate, *suffix)
                for stream in ("property_sample", "fold_partition")
            )
            out.append(
                (
                    (
                        draw.family,
                        draw.label,
                        replicate,
                        CALIBRATION_N,
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
    """Fit every property replication and return one row per cell and replication.

    ``record`` and ``cap`` exist for a harness check on the throwaway record of rule L9 alone.
    The declared run passes neither.
    """
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
    for column in INTEGER_COLUMNS:
        controls[column] = controls[column].astype(int)
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


def target_verdicts(summary: pd.DataFrame, *, floor: float) -> None:
    """The ``weighted_target_control`` verdicts: each control's coverage interval below ``floor``.

    The controls read the coverage of the unweighted :math:`\\Psi_{d_0}` and of the weighted
    value of the unweighted rule by the weighted fit, whose target is the weighted optimum, so
    each interval must miss.
    """
    family = summary["property"] == TARGET_FAMILY
    roles = set(summary.loc[family, "role"].astype(str))
    if family.any() and roles != {"control"}:
        raise ValueError(f"{TARGET_FAMILY} publishes roles {sorted(roles)}, not control")
    summary.loc[family, "passed"] = summary.loc[family, "coverage_ci_upper"] < floor


def rule_value_readings(group: pd.DataFrame) -> dict[str, float]:
    """One cell's readings against the value of the fit's own fold rules (``rule_value``).

    The error is ``estimate - rule_value``. The standardized bias is its mean over its SD, the
    coverage the share of draws with ``|error| <= CRITICAL * std_error``, and the SE ratio the
    mean SE over the error's SD. No verdict reads them.
    """
    error = group["estimate"].astype(float) - group["rule_value"].astype(float)
    std_error = group["std_error"].astype(float)
    spread = float(error.std())
    return {
        "rule_value_standardized_bias": float(error.mean()) / spread,
        "rule_value_coverage": float((error.abs() <= base.CRITICAL * std_error).mean()),
        "rule_value_se_ratio": float(std_error.mean()) / spread,
    }


def report_optimal(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish :data:`OPTIMAL_SUMMARY` for every cell."""
    for family in sorted(set(summary["property"])):
        groups = cell_groups(rows, family)
        for cell in summary.loc[summary["property"] == family, "cell"]:
            group = groups[cell]
            mask = (summary["property"] == family) & (summary["cell"] == cell)
            disagreement = group["rule_disagreement"].astype(float)
            summary.loc[mask, "mean_regret"] = float(group["regret"].astype(float).mean())
            summary.loc[mask, "mean_rule_disagreement"] = float(disagreement.mean())
            summary.loc[mask, "min_rule_disagreement"] = float(disagreement.min())
            summary.loc[mask, "max_rule_disagreement"] = float(disagreement.max())
            for column, value in rule_value_readings(group).items():
                summary.loc[mask, column] = value


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell. The study publishes no rate rows (``rate_labels=()``)."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=OPTIMAL_SUMMARY, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    target_verdicts(summary, floor=STUDY.margins.calibration_coverage[0])
    report_optimal(summary, rows)
    return finish(summary, rates)


def _coverage_floor(ctx: pf.Context) -> float:
    return ctx.margins.calibration_coverage[0]


#: The ``weighted_target_control`` rule of :func:`target_verdicts`.
TARGET_RULE = pf.FamilyRule(
    families=frozenset({TARGET_FAMILY}),
    shared="target_verdicts",
    clauses=(
        pf.proportion_clause(
            "unweighted-target coverage below the calibration floor",
            statistic="S2",
            judge=pf.below_judge(_coverage_floor),
            applies=pf.roles("control"),
        ),
    ),
)

#: The declared rules of the failure accounting: the shared calibration rule and the target rule.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None), TARGET_RULE)
