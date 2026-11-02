"""Repeated-sampling properties of the learned rule with an intermediate held at each level.

Two families, in the order the declaration reads them. Every fit is at n = 2,000 on the one law,
fold-evaluated, with ten folds and pooled targeting, and reports both levels of the intermediate.

``interval_calibration``
    The ``declared`` learned-rule fit at each level (``learned_z0__declared``,
    ``learned_z1__declared``) and the known rule ``1{W1 > 0}`` through the same path
    (``known_z0__declared``, ``known_z1__declared``), each with the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    :data:`SHRUNKEN_SE_FACTOR`, and noise of one efficiency-bound unit added to each estimate. Each
    level of both draws also has the ``term_dropped_control``: the same estimate with the SE of
    its curve rebuilt without :math:`1\\{Z = z\\}/q`, which must fall below the SE-ratio band.
``targeting_necessity``
    The ``drop_w1`` learner at each level: the targeted value against the untargeted fold average
    of the initial plug-ins.

Every statistic reads the error ``estimate - truth`` of each row, because the record declares
``truth_varies_by_replicate``; a known-rule row's truth is the rule's fixed value at its level. A
property draw's sample seed is ``stream_seed(record, "property_sample", family, label, replicate,
*suffix)`` and its fold seed the same tuple with ``"fold_partition"`` first; rule L3 of the RM30
studies sets each suffix. One draw writes both levels' rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_intermediate_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
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
    necessity_verdicts,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_intermediate import (
    CALIBRATION_LABELS,
    EFFICIENCY_BOUNDS,
    NECESSITY_ARMS,
    NECESSITY_FAMILY,
    NECESSITY_LABELS,
    POSITIVE,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PRIMARY_SCENARIO,
    PROPERTY_CELLS,
    STUDY,
    TERM_DROPPED,
)

base = law.base

CALIBRATION_N = PRIMARY_N
#: The learned draws of ``interval_calibration`` (both levels): the probe rule's fallback, fixed
#: before the pooled reading it applies to, sets the count at the run-time budget. The per-cell
#: rule (a product of at least 0.975 over the two levels) alone gives no count up to 40,000 on
#: the 4,000-draw pilot and 7,200 on the 8,000 draws pooled with the probe
#: (``test_learned_rule_intermediate_design``).
LEARNED_REPLICATES = 20_000
#: The known-rule draws of ``interval_calibration`` (both levels): the per-cell rule on the
#: 4,000-draw pilot (0.976). The arm-pooled rule returns no count for two arms on a pilot of this
#: size at any mean, and the fallback names the learned levels only.
KNOWN_REPLICATES = 10_000
#: The ``targeting_necessity`` draws: the smallest multiple of 400, at least 1,200, whose
#: predictive probability that both positives' bias legs, both controls' discrimination legs and
#: the displacement pass is at least 0.95 on the 4,000-draw pilot.
NECESSITY_REPLICATES = 1_200
#: The plan's shrunken-SE control of the calibration family: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80
#: The ``targeting_necessity`` displacement floor, in positive-arm SDs (the RM30 value).
TARGETING_DISPLACEMENT = 0.5

#: The draw labels.
LEARNED_DRAW = "learned"
KNOWN_DRAW = "known"
NECESSITY_DRAW = "drop_w1"

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    base.PropertyDraw("interval_calibration", LEARNED_DRAW, LEARNED_REPLICATES, PRIMARY_SCENARIO),
    base.PropertyDraw("interval_calibration", KNOWN_DRAW, KNOWN_REPLICATES, PRIMARY_SCENARIO),
    base.PropertyDraw(NECESSITY_FAMILY, NECESSITY_DRAW, NECESSITY_REPLICATES, PRIMARY_SCENARIO),
)

#: Each draw label's (outcome learner, known rule).
DRAW_SETTINGS: dict[str, tuple[str, bool]] = {
    LEARNED_DRAW: ("declared", False),
    KNOWN_DRAW: ("declared", True),
    NECESSITY_DRAW: ("drop_w1", False),
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

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, _ = key[0]
    return law.draw(n, sample_seed)


def draw_cells(label: str) -> list[str]:
    """Every ``property/cell`` one draw writes, its derived controls included."""
    if label == NECESSITY_DRAW:
        return [f"{NECESSITY_FAMILY}/{cell}" for cell in PROPERTY_CELLS[NECESSITY_FAMILY]]
    written = []
    for z in law.LEVELS:
        name = f"{label}_z{int(z)}"
        positive = f"interval_calibration/{name}__{POSITIVE}"
        written += [positive, *DERIVED_CELLS[positive]]
        written.append(f"interval_calibration/{name}__{TERM_DROPPED}")
    return written


def _cells_of(key: _Key) -> list[str]:
    return draw_cells(key[0][1])


def _positive(
    family: str, cell: str, where: tuple[int, int, int], truth: float, estimate: Any
) -> dict[str, Any]:
    replicate, n, requested = where
    return replicate_row(
        property_name=family,
        cell=cell,
        role="positive",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def _control(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    truth: float,
    estimate: float,
    standard_error: float,
) -> dict[str, Any]:
    replicate, n, requested = where
    return control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        standard_error=standard_error,
        critical=law.CRITICAL,
        role="control",
    )


def draw_rows(fitted: law.Measured, label: str, where: tuple[int, int, int]) -> list[dict]:
    """The rows one draw writes at both levels."""
    rows = []
    for z in law.LEVELS:
        estimate = fitted.estimate(z)
        truth = fitted.truths[z].truth
        columns = law.harness_columns(fitted, z)
        written = []
        if label == NECESSITY_DRAW:
            name = f"{NECESSITY_DRAW}_z{int(z)}"
            targeted, untargeted = (f"{name}__{arm}" for arm in NECESSITY_ARMS)
            written.append(_positive(NECESSITY_FAMILY, targeted, where, truth, estimate))
            written.append(
                _control(
                    NECESSITY_FAMILY,
                    untargeted,
                    where,
                    truth,
                    law.untargeted_estimate(fitted.results[z]),
                    float(estimate.std_error),
                )
            )
        else:
            name = f"{label}_z{int(z)}"
            written.append(
                _positive("interval_calibration", f"{name}__{POSITIVE}", where, truth, estimate)
            )
            written.append(
                _control(
                    "interval_calibration",
                    f"{name}__{TERM_DROPPED}",
                    where,
                    truth,
                    float(estimate.psi),
                    fitted.truths[z].dropped_std_error,
                )
            )
        for row in written:
            row.update(columns)
        rows += written
    return rows


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, label, replicate, n, requested, _, fold_seed, _ = key[0]
    if label not in DRAW_SETTINGS:
        raise ValueError(f"unknown property draw {label!r}")
    learner, known = DRAW_SETTINGS[label]
    fitted = law.measured(frame, learner, fold_seed, known=known)
    return draw_rows(fitted, label, (replicate, n, requested))


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
        critical=law.CRITICAL,
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


#: The ``targeting_necessity`` family's declaration, read by its verdict and its rule.
TARGETING_NECESSITY = property_verdicts.Necessity(
    family=NECESSITY_FAMILY,
    labels=NECESSITY_LABELS,
    arms=NECESSITY_ARMS,
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
    truth_varies=STUDY.truth_varies_by_replicate,
)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("targeting_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    return finish(summary, rates)


#: The declared rules of the failure accounting: the shared calibration rule, whose
#: ``term_dropped_control`` clause reads the SE ratio below the band, and the necessity rule.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
