"""Repeated-sampling properties of the fold-local learned-rule value CV-TMLE.

Four families, each on the ``non_exceptional`` law, in the order rule L3 reads them.

``interval_calibration``
    The declared fit at n = 2,000, 17,000 replications.  The positive cell's SE-ratio interval
    must lie inside 0.93 to 1.07 and its coverage interval inside 0.92 to 0.98.  Two controls
    are derived from its rows: the SE shrunken by 0.70, and noise of SD 0.664444 / sqrt(2000)
    added to each estimate.  Each control's SE-ratio interval must lie below 0.93.
``root_n_and_efficiency``
    The declared fit at n = 500 (control), 2,000 and 8,000, 6,000 replications each, with the
    framework rules and the ``root_n_rate`` slopes of the error SD and of the mean SE.
``targeting_necessity``
    An outcome learner that drops ``W1``, with the correct treatment learner, 2,655
    replications.  The positive arm is the fit.  The control is the same fit without its
    fluctuation: the fold average of the initial plug-ins at the fit's own rule.
``fold_locality``
    A random forest outcome learner, 2,655 replications.  The positive arm is the declared fit.
    The control is the deliberate mutation of
    :class:`~tests.studies._learned_rule_law.ValidationRowRule`, which learns each fold's rule
    on that fold's validation rows.  Each arm's truth is the value of the rules it used.  The
    family reads bias only.

Every statistic reads the error ``estimate - truth`` of each row, because the record declares
``truth_varies_by_replicate``.  A property draw's sample seed is
``stream_seed(record, "property_sample", family, label, replicate, *suffix)`` and its fold seed
the same tuple with ``"fold_partition"`` first.  Rule L3 sets each suffix: the calibration cell
moves to ``("retry", 1)``, and every other cell keeps its label.  Two paired arms share one
draw and one partition.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_law as law
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    property_role,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import stream_seed
from tests.studies.learned_rule_cvtmle import (
    EFFICIENCY_BOUND,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    STUDY,
)

LAW = "non_exceptional"
CALIBRATION_REPLICATES = 17_000
RATE_REPLICATES = 6_000
RATE_SIZES = (500, 2_000, 8_000)
NECESSITY_REPLICATES = 2_655
FOLD_LOCALITY_REPLICATES = 2_655
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.5
FOLD_LOCALITY_DISPLACEMENT = 0.5

#: The property draws in the order rule L3 reads them.  Each label is the one the declaration's
#: property table gives, before L3 appends a suffix.
PROPERTY_DRAWS: tuple[law.PropertyDraw, ...] = (
    law.PropertyDraw(
        "interval_calibration",
        f"{LAW}__correctly_specified",
        CALIBRATION_REPLICATES,
        LAW,
    ),
    *(
        law.PropertyDraw("root_n_and_efficiency", f"n_{size}", RATE_REPLICATES, LAW)
        for size in RATE_SIZES
    ),
    law.PropertyDraw("targeting_necessity", LAW, NECESSITY_REPLICATES, LAW),
    law.PropertyDraw("fold_locality", LAW, FOLD_LOCALITY_REPLICATES, LAW),
)

#: What rule L3 gives on the declared record, as "Seeds derived at the declaration" records.
DECLARED_SUFFIXES: dict[tuple[str, str], tuple[Any, ...]] = {
    (draw.family, draw.label): (("retry", 1) if draw.family == "interval_calibration" else ())
    for draw in PROPERTY_DRAWS
}


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The L3 suffix of every property draw of ``record``.

    On the declared record the result must be the declared one.  A difference means the
    framework's seed derivation moved after the declaration, and the run stops.
    """
    resolved = law.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)
    if record == STUDY and resolved != DECLARED_SUFFIXES:
        raise RuntimeError(
            f"rule L3 resolves the draw labels to {resolved}, not the declared "
            f"{DECLARED_SUFFIXES}; the declaration stops"
        )
    return resolved


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: law.Truth,
    estimate: Any,
    solver_warnings: int,
) -> dict[str, Any]:
    row = replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth.truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    row.update(
        {
            "oracle_se": truth.oracle_se,
            "rule_rows_checked": truth.rows_checked,
            "solver_warnings": solver_warnings,
        }
    )
    return row


def _replication(payload: tuple[str, str, int, int, int, int, int]) -> list[dict[str, Any]]:
    """Every row one property draw writes: one arm, or a paired pair on one sample."""
    family, label, replicate, n, requested, sample_seed, fold_seed = payload
    frame = law.draw(LAW, n, sample_seed)
    if family in ("interval_calibration", "root_n_and_efficiency"):
        result, truth, warnings = law.measured(frame, law.outcome_learner(), LAW, fold_seed)
        role = property_role(
            "declared",
            controls=(),
            property_name=family,
            n=n,
            rate_sizes=RATE_SIZES,
        )
        estimate = result.estimates[law.ESTIMAND]
        return [_row(family, label, role, replicate, n, requested, truth, estimate, warnings)]
    if family == "targeting_necessity":
        result, truth, warnings = law.measured(frame, law.drop_w1_learner(), LAW, fold_seed)
        estimate = result.estimates[law.ESTIMAND]
        targeted = _row(
            family,
            f"{LAW}__targeted",
            "positive",
            replicate,
            n,
            requested,
            truth,
            estimate,
            warnings,
        )
        untargeted = control_row(
            property_name=family,
            cell=f"{LAW}__untargeted",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth.truth,
            estimate=law.untargeted_estimate(result),
            standard_error=float(estimate.std_error),
            critical=law.CRITICAL,
        )
        untargeted.update({name: targeted[name] for name in law.HARNESS_COLUMNS})
        return [targeted, untargeted]
    if family == "fold_locality":
        rows = []
        for cell, role, mutation in (
            ("fold_local", "positive", False),
            ("validation_rows", "control", True),
        ):
            result, truth, warnings = law.measured(
                frame, law.forest_learner(), LAW, fold_seed, mutation=mutation
            )
            rows.append(
                _row(
                    family,
                    f"{LAW}__{cell}",
                    role,
                    replicate,
                    n,
                    requested,
                    truth,
                    result.estimates[law.ESTIMAND],
                    warnings,
                )
            )
        return rows
    raise ValueError(f"unknown property family {family!r}")


def _payloads(
    record: StudyRecord, cap: int | None
) -> list[tuple[tuple[str, str, int, int, int, int, int]]]:
    """One payload per property draw.  The slow forest draws go first, so the pool stays fed."""
    resolved = suffixes(record)
    order = sorted(PROPERTY_DRAWS, key=lambda draw: draw.family != "fold_locality")
    out = []
    for draw in order:
        suffix = resolved[draw.family, draw.label]
        n = (
            int(draw.label.removeprefix("n_"))
            if draw.family == "root_n_and_efficiency"
            else PRIMARY_N
        )
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            sample_seed = stream_seed(
                record, "property_sample", draw.family, draw.label, replicate, *suffix
            )
            fold_seed = stream_seed(
                record, "fold_partition", draw.family, draw.label, replicate, *suffix
            )
            out.append(
                ((draw.family, draw.label, replicate, n, replicates, sample_seed, fold_seed),)
            )
    return out


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY, cap: int | None = None
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication.

    ``record`` and ``cap`` exist for a harness check on the throwaway record of rule L9 alone.
    The declared run passes neither.
    """
    outcomes = map_parallel(_replication, _payloads(record, cap), n_jobs=n_jobs)
    rows = pd.DataFrame([row for records in outcomes for row in records])
    controls = calibration_controls(
        rows,
        record,
        labels=(LAW,),
        efficiency_bounds={LAW: EFFICIENCY_BOUND},
        calibration_n=PRIMARY_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=law.CRITICAL,
    )
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        if family == "root_n_rate":
            continue
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *law.HARNESS_COLUMNS]]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, every statistic on the error of each row."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "fold_locality_displacement"),
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=(LAW,),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
        truth_varies=STUDY.truth_varies_by_replicate,
    )
    necessity_verdicts(
        summary,
        rows,
        family="fold_locality",
        labels=(LAW,),
        arms=("fold_local", "validation_rows"),
        column="fold_locality_displacement",
        threshold=FOLD_LOCALITY_DISPLACEMENT,
        truth_varies=STUDY.truth_varies_by_replicate,
    )
    return finish(summary, rates)
