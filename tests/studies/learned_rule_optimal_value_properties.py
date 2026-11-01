"""Repeated-sampling properties of the value of the optimal rule.

Four families, in the order the declaration reads them. Every fit is fold-evaluated, with ten
folds and pooled targeting, and every cell reads the error against the fixed truth
:math:`\\Psi_{d_0}` of its law.

``interval_calibration``
    ``non_exceptional`` with the known ``g``, n = 2,000 (``non_exceptional_known_g__declared``),
    with the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    :data:`SHRUNKEN_SE_FACTOR`, and noise of one efficiency-bound unit added to each estimate.
``optimal_target_control``
    ``wrong_limit`` with the known ``g``, n = 2,500: the fit's own estimate and SE, whose
    learner's limit rule is not :math:`d_0`. Its exact coverage interval must lie wholly below
    the calibration floor.
``estimated_g_report``
    ``non_exceptional`` with the treatment model fitted, n = 2,000: role ``diagnostic``, no
    verdict.
``exceptional_law_report``
    ``partially_exceptional`` with the known ``g``, n = 2,000: role ``diagnostic``, no verdict.

Every cell's summary also publishes the mean regret :math:`\\Psi_{d_0} - \\tilde\\psi_{0n}` and
the mean, smallest and largest ``rule_disagreement`` over its draws (:data:`OPTIMAL_SUMMARY`),
read by no verdict. A property draw's sample seed is
``stream_seed(record, "property_sample", family, label, replicate, *suffix)`` and its fold seed
the same tuple with ``"fold_partition"`` first; rule L3 of the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_optimal_law as law
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
from tests.studies.learned_rule_optimal_value import (
    CALIBRATION_LABEL,
    CALIBRATION_LABELS,
    CONTROL_N,
    DIAGNOSTIC_FAMILIES,
    EFFICIENCY_BOUNDS,
    ESTIMATED,
    ESTIMATED_FAMILY,
    ESTIMATED_LABEL,
    EXCEPTIONAL,
    EXCEPTIONAL_FAMILY,
    EXCEPTIONAL_LABEL,
    EXCEPTIONAL_LAW,
    POSITIVE,
    PRIMARY_LAW,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STUDY,
    TARGET_CONTROL,
    TARGET_FAMILY,
    TARGET_LABEL,
    TARGET_LAW,
)

base = law.base

CALIBRATION_N = PRIMARY_N
#: The ``interval_calibration`` draws: the smallest multiple of 400 whose predictive probability
#: that the positive passes is at least 0.95, as the product of its SE-ratio leg (the per-arm
#: model of :func:`~tests.studies.mtp_study_tools.per_arm_pass`, one positive) and its coverage
#: leg (the exact interval inside the band), read on the first 4,000-draw pilot and on the 8,000
#: draws pooled with the probe, and the larger of the two (``test_learned_rule_optimal_design``).
#: The first pilot binds: SE ratio 0.971 and coverage 0.936 against :math:`\Psi_{d_0}`.
CALIBRATION_REPLICATES = 13_200
#: The ``optimal_target_control`` draws: the smallest multiple of 400, at least 1,000, whose
#: predictive probability that the control's coverage interval lies below the floor is at least
#: 0.95 on the 4,000-draw pilot; the floor binds (``test_learned_rule_optimal_design``).
TARGET_REPLICATES = 1_200
#: The ``estimated_g_report`` draws: the calibration count, so that the reported SE ratio and
#: coverage read at the precision of the gated cell. Declared as a rule; no reading moves it.
ESTIMATED_REPLICATES = CALIBRATION_REPLICATES
#: The ``exceptional_law_report`` draws: the study's floor of 1,000 draws for a cell, as a
#: multiple of 400. Declared as a rule; no reading moves it.
EXCEPTIONAL_REPLICATES = 1_200
#: The plan's shrunken-SE control of the calibration family: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    base.PropertyDraw(
        "interval_calibration", CALIBRATION_LABEL, CALIBRATION_REPLICATES, PRIMARY_LAW
    ),
    base.PropertyDraw(TARGET_FAMILY, TARGET_LABEL, TARGET_REPLICATES, TARGET_LAW),
    base.PropertyDraw(ESTIMATED_FAMILY, ESTIMATED_LABEL, ESTIMATED_REPLICATES, PRIMARY_LAW),
    base.PropertyDraw(
        EXCEPTIONAL_FAMILY, EXCEPTIONAL_LABEL, EXCEPTIONAL_REPLICATES, EXCEPTIONAL_LAW
    ),
)


#: Each family's setting: (law, known ``g``, n, the cell suffix, the role).
SETTINGS: dict[str, tuple[str, bool, int, str, str]] = {
    "interval_calibration": (PRIMARY_LAW, True, CALIBRATION_N, POSITIVE, "positive"),
    TARGET_FAMILY: (TARGET_LAW, True, CONTROL_N, TARGET_CONTROL, "control"),
    ESTIMATED_FAMILY: (PRIMARY_LAW, False, CALIBRATION_N, ESTIMATED, DIAGNOSTIC_ROLE),
    EXCEPTIONAL_FAMILY: (EXCEPTIONAL_LAW, True, CALIBRATION_N, EXCEPTIONAL, DIAGNOSTIC_ROLE),
}


#: Each family's draw label.
LABELS = {
    "interval_calibration": CALIBRATION_LABEL,
    TARGET_FAMILY: TARGET_LABEL,
    ESTIMATED_FAMILY: ESTIMATED_LABEL,
    EXCEPTIONAL_FAMILY: EXCEPTIONAL_LABEL,
}


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arm to the controls derived from its rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = law.HARNESS_COLUMNS

#: The integer harness columns, which a derived control must keep integer.
INTEGER_COLUMNS = ("rule_rows_checked", "solver_warnings")

#: The descriptive summary columns of every cell: the mean regret, and the mean, smallest and
#: largest ``rule_disagreement`` over its draws. No verdict reads them.
OPTIMAL_SUMMARY = (
    "mean_regret",
    "mean_rule_disagreement",
    "min_rule_disagreement",
    "max_rule_disagreement",
)

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, scenario = key[0]
    return law.draw(scenario, n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    family = key[0][0]
    if family == "interval_calibration":
        # A lost positive draw is lost to its derived controls too.
        positive = f"{family}/{CALIBRATION_LABEL}__{POSITIVE}"
        return [positive, *DERIVED_CELLS[positive]]
    return [f"{family}/{cell}" for cell in PROPERTY_CELLS[family]]


def _row(family: str, where: tuple[int, int, int], fitted: law.Measured) -> dict[str, Any]:
    scenario, _, _, suffix, role = SETTINGS[family]
    replicate, n, requested = where
    cell = f"{LABELS[family]}__{suffix}"
    estimate = fitted.result.estimates[law.ESTIMAND]
    truth = law.law_limit(scenario)["truth"]
    if role == "control":
        row = control_row(
            property_name=family,
            cell=cell,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth,
            estimate=float(estimate.psi),
            standard_error=float(estimate.std_error),
            critical=base.CRITICAL,
            role=role,
        )
    else:
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
    row.update(law.harness_columns(fitted, scenario))
    return row


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, _, replicate, n, requested, _, fold_seed, scenario = key[0]
    if family not in SETTINGS:
        raise ValueError(f"unknown property family {family!r}")
    declared_law, known_g, declared_n, _, _ = SETTINGS[family]
    if (scenario, n) != (declared_law, declared_n):
        raise ValueError(f"{family} draws {declared_law} at n = {declared_n}, not {scenario}, {n}")
    fitted = law.measured(frame, scenario, law.LEARNERS[scenario](), fold_seed, known_g=known_g)
    return [_row(family, (replicate, n, requested), fitted)]


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        n = SETTINGS[draw.family][2]
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
                        n,
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
    """The ``optimal_target_control`` verdict: the control's coverage interval below ``floor``.

    The control reads the coverage of :math:`\\Psi_{d_0}` by a fit whose learner's limit rule
    is not :math:`d_0`, so the regret is of the order of the SE and the interval must miss.
    """
    family = summary["property"] == TARGET_FAMILY
    roles = set(summary.loc[family, "role"].astype(str))
    if family.any() and roles != {"control"}:
        raise ValueError(f"{TARGET_FAMILY} publishes roles {sorted(roles)}, not control")
    summary.loc[family, "passed"] = summary.loc[family, "coverage_ci_upper"] < floor


def report_optimal(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish :data:`OPTIMAL_SUMMARY` for every cell, and the diagnostic families' verdicts.

    A diagnostic cell's summary already holds its bias, coverage and SE ratio against
    :math:`\\Psi_{d_0}`. ``passed`` and ``property_passed`` read ``True`` there, meaning the
    numbers were computed and are published; the published role says that no margin was read.
    """
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
    for family in DIAGNOSTIC_FAMILIES:
        selected = summary["property"] == family
        roles = set(summary.loc[selected, "role"].astype(str))
        if selected.any() and roles != {DIAGNOSTIC_ROLE}:
            raise ValueError(f"{family} publishes roles {sorted(roles)}, not diagnostic")
        summary.loc[selected, "passed"] = True
        summary.loc[selected, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, and the report of the diagnostic families.

    The study publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=OPTIMAL_SUMMARY, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    target_verdicts(summary, floor=STUDY.margins.calibration_coverage[0])
    report_optimal(summary, rows)
    return finish(summary, rates)


def _coverage_floor(ctx: pf.Context) -> float:
    return ctx.margins.calibration_coverage[0]


#: The ``optimal_target_control`` rule of :func:`target_verdicts`.
TARGET_RULE = pf.FamilyRule(
    families=frozenset({TARGET_FAMILY}),
    shared="target_verdicts",
    clauses=(
        pf.proportion_clause(
            "wrong-limit coverage below the calibration floor",
            statistic="S2",
            judge=pf.below_judge(_coverage_floor),
            applies=pf.roles("control"),
        ),
    ),
)

#: The declared rules of the failure accounting: the shared calibration rule and the target
#: rule. The diagnostic families write their verdict columns without reading a margin, so they
#: have none.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None), TARGET_RULE)
