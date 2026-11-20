"""The property family of ``score-assessment-compositions``, and its verdicts.

Every group of :data:`tests.studies.score_assessment_compositions.GROUPS` is one paired score
assessment per replication, and its rows fill the ``interval_calibration`` cells
:func:`~tests.studies.score_assessment_compositions.group_cells` names: each row of the group's
risk arm and calibration arm, the ``shrunken_se_control`` on the weighted S risk difference (the
reported SE times 0.70), and the ``row_variance_control`` on the clustered S risk difference and
S calibration slope (the row-level SE that ignores the clusters).  Both controls must fall below
the band.

Every draw names its stream (:data:`STREAMS`): the declared run reads ``property``, the pilot
``pilot`` and its extension ``pilot_2``, and a smoke ``smoke``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import score_assessment_compositions as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS

#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The sample and fold labels of each stream.  ``property`` is the declared run.
STREAMS: dict[str, tuple[str, str]] = {
    "property": ("property_sample", "fold_partition"),
    "pilot": ("pilot_sample", "pilot_fold_partition"),
    "pilot_2": ("pilot_2_sample", "pilot_2_fold_partition"),
    "smoke": ("smoke_sample", "smoke_fold_partition"),
}

#: The positive calibration kinds, one per gated arm (the cell suffix after ``__``).
POSITIVE_ARMS: tuple[str, ...] = tuple(arm.removeprefix("__") for arm in study.CALIBRATION_ARMS)


def sample_seed(label: str, replicate: int, stream: str = "property") -> int:
    """The sample seed of replication ``replicate`` of a group on ``stream``."""
    return stream_seed(STUDY, STREAMS[stream][0], label, replicate)


def fold_seed(label: str, replicate: int, stream: str = "property") -> int:
    """The fold seed of replication ``replicate`` of a group on ``stream``."""
    return stream_seed(STUDY, STREAMS[stream][1], label, replicate)


def _draw(label: str, replicate: int, stream: str) -> Any:
    group = study.GROUP_BY_LABEL[label]
    return study.draw_from_seed(label, group.n, sample_seed(label, replicate, stream))


def _replicate(label: str, replicate: int, stream: str, drawn: Any) -> list[dict[str, Any]]:
    """Every property row of one replication of one group."""
    group = study.GROUP_BY_LABEL[label]
    frame, truths = drawn
    fitted = study.fit_group(group, frame, random_state=fold_seed(label, replicate, stream))
    rows = [
        replicate_row(
            property_name=study.CALIBRATION,
            cell=study.calibration_cell(group, arm, row),
            role="positive",
            replicate=replicate,
            n=group.n,
            requested=group.replicates,
            truth=float(truths[row]),
            estimate=fitted["estimates"][row],
            alpha=STUDY.margins.alpha,
        )
        for arm in group.arms
        for row in group.rows(arm)
    ]
    for control, labels in group.controls.items():
        for row in labels:
            rows.append(
                control_row(
                    property_name=study.CALIBRATION,
                    cell=study.control_cell(group, control, row),
                    replicate=replicate,
                    n=group.n,
                    requested=group.replicates,
                    truth=float(truths[row]),
                    estimate=float(fitted["estimates"][row].psi),
                    standard_error=float(fitted["controls"][control][row]),
                    critical=study.CRITICAL,
                )
            )
    return rows


def _cells_of(key: tuple[str, int, str]) -> list[str]:
    group = study.GROUP_BY_LABEL[key[0]]
    return [
        f"{family}/{cell}" for family, cells in study.group_cells(group).items() for cell in cells
    ]


def _cell_of(key: tuple[str, int, str]) -> str:
    return f"group/{key[0]}"


def _replicate_of(key: tuple[str, int, str]) -> int:
    return key[1]


def payloads(budget: int | None = None, stream: str = "property") -> list[tuple[str, int, str]]:
    """Every ``(group, replicate, stream)`` key, in group order; ``budget`` caps each group."""
    return [
        (group.label, index, stream)
        for group in study.GROUPS
        for index in range(group.replicates if budget is None else min(budget, group.replicates))
    ]


def generate_property_rows(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    groups: tuple[str, ...] | None = None,
    stream: str = "property",
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each group and ``groups`` picks some.

    A run of every group must fill exactly the declared cells; a run over ``groups`` checks
    only that each row it wrote is declared.
    """
    keys = [key for key in payloads(budget, stream) if groups is None or key[0] in groups]
    outcomes = fit_properties(
        keys,
        splat_drawn(_replicate),
        draw=splat_draw(_draw),
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        seed_of=lambda key: sample_seed(*key),
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    if budget is not None:
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    declared = {(family, cell) for family, cells in PROPERTY_CELLS.items() for cell in cells}
    found = set(zip(rows["property"], rows["cell"], strict=True))
    if found - declared or (groups is None and found != declared):
        raise RuntimeError(f"cells {sorted(found ^ declared)} differ from the declaration")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared calibration verdicts of every positive arm and control."""
    summary, rates = apply_shared_verdicts(
        rows.loc[:, list(REPLICATE_COLUMNS)], STUDY, rate_labels=()
    )
    calibration_verdicts(
        summary, margins=STUDY.margins, efficiency_band=None, positive_suffix=POSITIVE_ARMS
    )
    return finish(summary, rates)


#: The failure accounting's declared rules.  The calibration family reads the shared rules
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES: tuple[Any, ...] = ()
