"""Repeated-sampling properties of the learned two-node regimen of ``LTMLE``.

Four families, in the order the declaration reads them. Every fit is at n = 2,000 on the one law,
cross-fitted with ten folds.

``interval_calibration``
    The ``declared`` fit (``two_node__declared``), with the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    :data:`SHRUNKEN_SE_FACTOR`, and noise of one efficiency-bound unit added to each estimate.
``conservative_coverage``
    The ``squared_w1`` node-1 learner, held to the bias leg and to a coverage interval whose
    lower bound is at or above the calibration floor, with no SE-ratio gate. Its control is the
    same estimate with its SE times
    :data:`~tests.studies.learned_regimen_ltmle.COVERAGE_SHRINK`, whose coverage interval must
    lie wholly below the floor.
``targeting_necessity``
    Two labels, each with its own draws: the ``squared_w1`` node-1 learner and the declared
    learners (``two_node``), each the targeted value against the untargeted plug-in, the mean of
    the estimator's stitched node-1 initial values.
``fold_locality``
    The declared learners twice on one sample and fold seed: the fit against the same fit with
    each fold's rules learned on its validation rows.

Every statistic reads the error ``estimate - truth`` of each row, because the record declares
``truth_varies_by_replicate``; each arm of ``fold_locality`` subtracts the value of its own rule
pairs. A property draw's sample seed is ``stream_seed(record, "property_sample", family, label,
replicate, *suffix)`` and its fold seed the same tuple with ``"fold_partition"`` first; rule L3 of
the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_regimen_law as law
from tests.studies.evidence import property_failures as pf
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
from tests.studies.learned_regimen_ltmle import (
    CALIBRATION_LABELS,
    COVERAGE_FAMILY,
    COVERAGE_LABEL,
    COVERAGE_SHRINK,
    EFFICIENCY_BOUNDS,
    LOCALITY_ARMS,
    LOCALITY_FAMILY,
    LOCALITY_LABEL,
    NECESSITY_ARMS,
    NECESSITY_FAMILY,
    NECESSITY_LABELS,
    POSITIVE,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PRIMARY_SCENARIO,
    PROPERTY_CELLS,
    STUDY,
)

base = law.base

CALIBRATION_N = PRIMARY_N
#: The ``interval_calibration`` draws: the per-cell rule (the positive's SE-ratio leg times its
#: coverage leg, at least 0.95) on the pooled 8,000 pilot draws (0.952). The probe rule's budget
#: fallback, measured on a host at load 8, gave 4,000; the run host's timing admits 9,200 within
#: the same 75-minute budget (design review, finding 1), on the timing condition the design
#: record states (``test_learned_regimen_ltmle_design``).
CALIBRATION_REPLICATES = 9_200
#: The ``conservative_coverage`` draws: the smallest multiple of 400, at least 1,200, whose
#: predictive probability that the positive's bias leg and coverage floor and the control's
#: coverage ceiling pass is at least 0.95 on the 4,000-draw redesign pilot of ``squared_w1``.
COVERAGE_REPLICATES = 2_400
#: The ``targeting_necessity`` draws of each label: the same rule over the positive's bias leg,
#: the control's discrimination leg and the displacement, on the 4,000 ``squared_w1`` draws of the
#: redesign pilot and the 8,000 pooled ``declared`` draws. Each label has its own draws.
NECESSITY_REPLICATES = 1_200
#: The ``fold_locality`` draws, two fits each: the same rule on the 4,000 paired draws.
LOCALITY_REPLICATES = 1_200
#: The plan's shrunken-SE control of the calibration family: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80
#: The displacement floors, in positive-arm SDs (the RM30 values).
TARGETING_DISPLACEMENT = 0.5
FOLD_LOCALITY_DISPLACEMENT = 0.5

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    base.PropertyDraw(
        "interval_calibration", CALIBRATION_LABELS[0], CALIBRATION_REPLICATES, PRIMARY_SCENARIO
    ),
    base.PropertyDraw(COVERAGE_FAMILY, COVERAGE_LABEL, COVERAGE_REPLICATES, PRIMARY_SCENARIO),
    *(
        base.PropertyDraw(NECESSITY_FAMILY, label, NECESSITY_REPLICATES, PRIMARY_SCENARIO)
        for label in NECESSITY_LABELS
    ),
    base.PropertyDraw(LOCALITY_FAMILY, LOCALITY_LABEL, LOCALITY_REPLICATES, PRIMARY_SCENARIO),
)

#: Each property draw's node-1 learner, by ``(family, label)``.
DRAW_LEARNERS: dict[tuple[str, str], str] = {
    ("interval_calibration", CALIBRATION_LABELS[0]): "declared",
    (COVERAGE_FAMILY, COVERAGE_LABEL): COVERAGE_LABEL,
    (NECESSITY_FAMILY, "squared_w1"): "squared_w1",
    (NECESSITY_FAMILY, law.LAW): "declared",
    (LOCALITY_FAMILY, LOCALITY_LABEL): "declared",
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


def _label_cells(family: str, label: str) -> list[str]:
    """The declared cells of ``family`` that belong to ``label``, in declaration order."""
    return [cell for cell in PROPERTY_CELLS[family] if cell.startswith(f"{label}__")]


def draw_cells(family: str, label: str) -> list[str]:
    """Every ``property/cell`` one draw of ``(family, label)`` writes, its derived controls
    included."""
    written = [f"{family}/{cell}" for cell in _label_cells(family, label)]
    if family == "interval_calibration":
        positive = f"{family}/{CALIBRATION_LABELS[0]}__{POSITIVE}"
        written = [positive, *DERIVED_CELLS[positive]]
    return written


def _cells_of(key: _Key) -> list[str]:
    return draw_cells(key[0][0], key[0][1])


def _positive(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    fitted: law.Measured,
    role: str = "positive",
) -> dict[str, Any]:
    """A row of one fit's own estimate; ``role`` is ``control`` for the mutated
    ``fold_locality`` arm, which reports its own estimate and SE."""
    replicate, n, requested = where
    row = replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=fitted.truth.truth,
        estimate=fitted.estimate,
        alpha=STUDY.margins.alpha,
    )
    row.update(law.harness_columns(fitted))
    return row


def _control(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    fitted: law.Measured,
    estimate: float,
    standard_error: float,
) -> dict[str, Any]:
    replicate, n, requested = where
    row = control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=fitted.truth.truth,
        estimate=estimate,
        standard_error=standard_error,
        critical=law.CRITICAL,
        role="control",
    )
    row.update(law.harness_columns(fitted))
    return row


def draw_rows(
    family: str, label: str, frame: pd.DataFrame, fold_seed: int, where: tuple[int, int, int]
) -> list[dict[str, Any]]:
    """The rows one draw of ``(family, label)`` writes."""
    learner = DRAW_LEARNERS[family, label]
    if family == LOCALITY_FAMILY:
        local, mutated = (
            law.measured(frame, learner, fold_seed, mutation=mutation) for mutation in (False, True)
        )
        positive_cell, control_cell = _label_cells(family, label)
        return [
            _positive(family, positive_cell, where, local),
            _positive(family, control_cell, where, mutated, role="control"),
        ]
    fitted = law.measured(frame, learner, fold_seed)
    point, std_error = float(fitted.estimate.psi), float(fitted.estimate.std_error)
    if family == "interval_calibration":
        return [_positive(family, f"{CALIBRATION_LABELS[0]}__{POSITIVE}", where, fitted)]
    positive_cell, control_cell = _label_cells(family, label)
    if family == COVERAGE_FAMILY:
        control_point, control_se = point, COVERAGE_SHRINK * std_error
    elif family == NECESSITY_FAMILY:
        control_point, control_se = law.untargeted_estimate(fitted.result), std_error
    else:
        raise ValueError(f"unknown property family {family!r}")
    return [
        _positive(family, positive_cell, where, fitted),
        _control(family, control_cell, where, fitted, control_point, control_se),
    ]


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, n, requested, _, fold_seed, _ = key[0]
    return draw_rows(family, label, frame, fold_seed, (replicate, n, requested))


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    """One payload per property draw. The paired ``fold_locality`` draws go first, so the pool
    stays fed."""
    resolved = suffixes(record)
    order = sorted(PROPERTY_DRAWS, key=lambda draw: draw.family != LOCALITY_FAMILY)
    out = []
    for draw in order:
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

#: The ``fold_locality`` family's declaration, read by its verdict and its rule.
FOLD_LOCALITY = property_verdicts.Necessity(
    family=LOCALITY_FAMILY,
    labels=(LOCALITY_LABEL,),
    arms=LOCALITY_ARMS,
    column="fold_locality_displacement",
    threshold=FOLD_LOCALITY_DISPLACEMENT,
    truth_varies=STUDY.truth_varies_by_replicate,
)


def coverage_verdicts(summary: pd.DataFrame, *, floor: float) -> None:
    """The ``conservative_coverage`` verdicts: a one-sided coverage claim, and its control.

    A positive passes when its bias is established inside the margin and the lower bound of its
    exact coverage interval is at or above ``floor``; its SE ratio is published and read by no
    verdict, because the claim is one-sided: it admits over-coverage, and it does not assert that
    the interval is conservative. A control passes when its coverage interval lies wholly below
    ``floor``. The rule is ``learned-rule-contrast``'s.
    """
    family = summary["property"] == COVERAGE_FAMILY
    positive = family & (summary["role"] == "positive")
    control = family & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"].astype(bool) & (
        summary.loc[positive, "coverage_ci_lower"] >= floor
    )
    summary.loc[control, "passed"] = summary.loc[control, "coverage_ci_upper"] < floor


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "fold_locality_displacement"),
        rate_labels=(),
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    coverage_verdicts(summary, floor=STUDY.margins.calibration_coverage[0])
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    necessity_verdicts(summary, rows, FOLD_LOCALITY)
    return finish(summary, rates)


def _coverage_floor(ctx: pf.Context) -> float:
    return ctx.margins.calibration_coverage[0]


#: The ``conservative_coverage`` rule of :func:`coverage_verdicts`.
COVERAGE_RULE = pf.FamilyRule(
    families=frozenset({COVERAGE_FAMILY}),
    shared="coverage_verdicts",
    clauses=(
        pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
        pf.proportion_clause(
            "coverage at or above the calibration floor",
            statistic="S2",
            judge=pf.floor_judge(_coverage_floor),
            applies=pf.roles("positive"),
        ),
        pf.proportion_clause(
            "shrunken coverage below the calibration floor",
            statistic="S2",
            judge=pf.below_judge(_coverage_floor),
            applies=pf.roles("control"),
        ),
    ),
)

#: The declared rules of the failure accounting: the shared calibration rule, the coverage rule
#: and the two necessity rules.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    COVERAGE_RULE,
    *property_verdicts.necessity_rules(TARGETING_NECESSITY, FOLD_LOCALITY),
)
