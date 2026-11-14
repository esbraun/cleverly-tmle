r"""The ``few_cluster_reference`` family of ``few-cluster-floor-cross-fitted-ltmle``.

Each key fits one draw of a ``(law, J)`` cell
(``stream_seed(STUDY, "property", f"{law}/J{J}", replicate)``; the pilot reads ``"pilot"``) and
writes one row per arm (:func:`tests.studies.few_cluster_floor_ltmle.fit_rows`). A fit that
raises is a lost key, a miss under the failure accounting.

The gated rule is the registered two-clause rule of ``clustered-few-cluster-tmle``
(:func:`~tests.studies.evidence.property_verdicts.few_cluster_reference_verdicts`): G1 and G2 on
``t_reference``, both control clauses on ``iid_t_control``. The readings are
:func:`tests.studies.few_cluster_floor.floor_readings`. :func:`family_floor` reads the family's
floor and :func:`decision` the part (c) comparison on the first 2,000 draws of each cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_ltmle as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import (
    FEW_CLUSTER_COLUMNS,
    apply_shared_verdicts,
    few_cluster_reference_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAMILY: Final[str] = study.FAMILY

#: One key per draw: ``(law, J, replicate, budget, stream)``.
Key = tuple[str, int, int, "int | None", str]

#: The arms of one size law and count, which pair the control with its positive arm.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    FAMILY: tuple(
        tuple(study.cell_name(sizes, clusters, arm) for arm, _ in study.arms(clusters))
        for sizes, clusters in study.grid()
    )
}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its role, size and requested replicates."""
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=study.cell_name(sizes, clusters, arm),
            dgp=(sizes, clusters),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=study.CLUSTER_SIZE * clusters,
            replicates=study.requested(arm),
            seed=study.draw_seed(sizes, clusters, 0),
            role=role,
            estimand=study.TARGET,
        )
        for sizes, clusters in study.grid()
        for arm, role in study.arms(clusters)
    )


def _draw(key: Key) -> tuple[pd.DataFrame, dict[str, float]]:
    sizes, clusters, replicate, _, stream = key
    return study.draw(sizes, clusters, study.draw_seed(sizes, clusters, replicate, stream))


def _cells_of(key: Key) -> list[str]:
    sizes, clusters, replicate, _, _ = key
    return [
        f"{FAMILY}/{study.cell_name(sizes, clusters, arm)}"
        for arm, _ in study.arms(clusters)
        if arm != floor.JACKKNIFE_ARM or replicate < floor.CANDIDATE_DRAWS
    ]


def _cell_of(key: Key) -> str:
    sizes, clusters, _, _, _ = key
    return f"{FAMILY}/{study.cell_name(sizes, clusters, floor.REFERENCE_ARM)}"


def _replicate_of(key: Key) -> int:
    return key[2]


def _fit_key(key: Key, drawn: tuple[pd.DataFrame, dict[str, float]]) -> list[dict[str, Any]]:
    sizes, clusters, replicate, budget, _ = key
    return study.fit_rows(sizes, clusters, replicate, drawn, budget=budget)


def keys(budget: int | None = None, stream: str = "property") -> list[Key]:
    """Every key of the grid, each cell's replicates capped at ``budget``."""
    return [
        (sizes, clusters, replicate, budget, stream)
        for sizes, clusters in study.grid()
        for replicate in range(study.requested(floor.REFERENCE_ARM, budget))
    ]


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, stream: str = "property"
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pilot or a smoke."""
    outcomes = fit_properties(
        keys(budget, stream),
        _fit_key,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    return rows.reindex(
        columns=[*REPLICATE_COLUMNS, *floor.EXTRA_COLUMNS, *floor.PACKAGE_COLUMNS]
    ).sort_values(["property", "cell", "replicate"], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the registered few-cluster rule and the floor readings."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=(*FEW_CLUSTER_COLUMNS, *floor.READING_COLUMNS), rate_labels=()
    )
    few_cluster_reference_verdicts(summary, rows, STUDY, PROPERTY_GROUPS[FAMILY])
    floor.floor_readings(summary, rows, STUDY)
    return finish(summary, rates)


def family_floor(summary: pd.DataFrame) -> int:
    """The family's floor under the 5.8 rule: every count passes on both size laws."""
    passed = summary.loc[summary["property"] == FAMILY].set_index("cell")["passed"].astype(bool)
    verdicts = {
        (study.FIT, sizes, clusters): bool(
            passed.get(study.cell_name(sizes, clusters, floor.REFERENCE_ARM), False)
            and passed.get(study.cell_name(sizes, clusters, "iid_t_control"), False)
        )
        for sizes, clusters in study.grid()
    }
    return floor.family_floor(floor.count_passes(verdicts, [study.FIT]), anchor=study.ANCHOR)


def candidate_cells(
    rows: pd.DataFrame, *, draws: int = floor.CANDIDATE_DRAWS
) -> list[floor.CandidateCell]:
    """The part (c) readings of every cell on its first ``draws`` draws (M3)."""
    family_rows = rows.loc[rows["property"] == FAMILY]
    groups = dict(tuple(family_rows.groupby("cell", sort=False)))
    out = []
    for sizes, clusters in study.grid():
        reference = groups.get(study.cell_name(sizes, clusters, floor.REFERENCE_ARM))
        if reference is None:  # every draw of the cell was lost: no candidate reads it
            continue
        for candidate in floor.CANDIDATE_ARMS:
            name = study.cell_name(sizes, clusters, candidate)
            frame = groups.get(name, reference.iloc[0:0])
            first = frame.loc[frame["replicate"] < draws]
            shown = first.loc[first["withheld"] == 0]
            length, high = 1.0, 1.0
            if candidate != floor.REFERENCE_ARM and len(shown) >= 2:
                length, interval = floor.length_ratio(
                    frame,
                    reference,
                    draws=draws,
                    replicates=STUDY.margins.bootstrap_replicates,
                    seed=stream_seed(STUDY, "length", name),
                )
                high = interval.high
            out.append(
                floor.CandidateCell(
                    candidate=candidate,
                    configuration=study.FIT,
                    law=sizes,
                    clusters=clusters,
                    pair=floor.paired_counts(frame, reference, draws=draws),
                    hits=int(shown["covered"].sum()),
                    shown=len(shown),
                    length=float(length),
                    length_high=float(high),
                )
            )
    return out


def decision(summary: pd.DataFrame, rows: pd.DataFrame) -> floor.Decision:
    """The part (c) decision of the family (plan 5.7)."""
    return floor.reference_decision(
        candidate_cells(rows),
        floor_a=family_floor(summary),
        jackknife_start=min(study.jackknife_counts()),
    )


def sizing(summary: pd.DataFrame) -> floor.FamilySizing:
    """The family's M5 sizing off a pilot summary (``generate_property_rows(stream="pilot")``)."""
    table = summary.loc[summary["property"] == FAMILY].set_index("cell")
    cells = {}
    for sizes, clusters in study.grid():
        positive = study.cell_name(sizes, clusters, floor.REFERENCE_ARM)
        control = study.cell_name(sizes, clusters, "iid_t_control")
        if positive in table.index and control in table.index:
            cells[(study.FIT, sizes, clusters)] = floor.sizing_inputs(
                table.loc[positive], table.loc[control]
            )
    return floor.family_sizing(cells, anchor=study.ANCHOR)


def synthetic_columns(cell: Any, rng: Any) -> dict[str, Any]:
    """The extra columns of a synthetic row: a shown interval.

    For the property differential once the record joins it. It is not a subject before the
    run: its synthetic rows at the declared budget hold about four million rows here, more
    than the fast tier can summarize.
    """
    del rng
    _, clusters = cell.dgp
    return {
        "withheld": 0,
        "trigger": "",
        "reference_df": float(clusters - 2),
        "clusters": clusters,
        "half_width": float("nan"),
        "failed_refits": 0,
    }


#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS
#: The failure accounting's bound (:data:`tests.studies.few_cluster_floor.FAILURE_BOUND`).
FAILURE_BOUND = floor.FAILURE_BOUND


__all__: Sequence[str] = (
    "PROPERTY_GROUPS",
    "candidate_cells",
    "decision",
    "declared_cells",
    "family_floor",
    "generate_property_rows",
    "sizing",
    "summarize_properties",
)
