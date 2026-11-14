r"""The ``few_cluster_reference`` family of ``few-cluster-floor-tmle``.

Each key fits one fit of :data:`tests.studies.few_cluster_floor_tmle.FITS` on one draw of a
``(law, J)`` cell and writes one row per arm
(:func:`tests.studies.few_cluster_floor_tmle.fit_rows`). Every fit of a ``(law, J, replicate)``
reads the same sample, ``stream_seed(STUDY, "property", f"{law}/J{J}", replicate)``; the pilot
reads the ``"pilot"`` stream. A fit that raises is a lost key: it costs that fit's cells one
replicate and no other fit's, and the failure accounting counts it as a miss.

**Gated** (the registered two-clause rule of ``clustered-few-cluster-tmle``,
:func:`~tests.studies.evidence.property_verdicts.few_cluster_reference_verdicts`): on each
``t_reference`` row, G1 (the exact 99% lower end of coverage over every draw, withheld rows as
misses and lost draws by the accounting's worst case, at least 0.90) and G2 (the bias within
0.25 empirical SD over every draw with a point); on each ``iid_t_control`` row, the paired
IID/cluster SE ratio's 99% upper end below 0.80 and the positive arm's SE-ratio upper end at
most 1.20. Every other arm is reported.

**Readings** (:func:`tests.studies.few_cluster_floor.floor_readings`): shown and withheld counts, the
withheld share by trigger, shown-only coverage with its exact interval, all-draw coverage with
its exact interval, the shown-only SE ratio, the over-coverage reading against 0.99, and on an
``ltmle_in_sample`` positive row the package-GLM refit's raised share with its exact interval
and the study learner's shown coverage split by whether the package GLM raised (M2, Q15).

**Floors and part (c).** :func:`family_floors` reads each family's floor off the verdicts
(:func:`tests.studies.few_cluster_floor.family_floor`); :func:`candidate_cells` and
:func:`decisions` read the part (c) comparison on the first 2,000 draws of each cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_tmle as study
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

#: One key per fit of a draw: ``(law, J, replicate, fit, budget, stream)``.
Key = tuple[str, int, int, str, "int | None", str]


def _groups() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(study.cell_name(fit, law, clusters, arm) for arm, _ in study.arms(fit, clusters))
        for fit, law, clusters in study.grid()
    )


#: The arms of one fit, size law and count, which pair each control with its positive arm.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {FAMILY: _groups()}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its role, size and requested replicates."""
    out = []
    for fit, law, clusters in study.grid():
        for arm, role in study.arms(fit, clusters):
            out.append(
                PropertyCell(
                    property=FAMILY,
                    cell=study.cell_name(fit, law, clusters, arm),
                    dgp=(law, clusters),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=10 * clusters,
                    replicates=study.requested(fit, arm),
                    seed=study.draw_seed(law, clusters, 0),
                    role=role,
                    estimand=study.LTMLE_TARGET
                    if study.FITS[fit] == "ltmle_in_sample"
                    else study.TARGET,
                )
            )
    return tuple(out)


def _draw(key: Key) -> pd.DataFrame:
    """The sample of a key, a function of ``(law, J, replicate, stream)`` alone."""
    law, clusters, replicate, _, _, stream = key
    return study.draw(law, clusters, study.draw_seed(law, clusters, replicate, stream))


def _cells_of(key: Key) -> list[str]:
    law, clusters, replicate, fit, _, _ = key
    return [
        f"{FAMILY}/{study.cell_name(fit, law, clusters, arm)}"
        for arm, _ in study.arms(fit, clusters)
        if arm != floor.JACKKNIFE_ARM or replicate < floor.CANDIDATE_DRAWS
    ]


def _cell_of(key: Key) -> str:
    law, clusters, _, fit, _, _ = key
    return f"{FAMILY}/{study.cell_name(fit, law, clusters, floor.REFERENCE_ARM)}"


def _replicate_of(key: Key) -> int:
    return key[2]


def _fit_key(key: Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    law, clusters, replicate, fit, budget, stream = key
    return study.fit_rows(fit, law, clusters, replicate, frame, budget=budget, stream=stream)


def keys(budget: int | None = None, stream: str = "property") -> list[Key]:
    """Every key of the grid, each fit's replicates capped at ``budget``."""
    out: list[Key] = []
    for fit, law, clusters in study.grid():
        for replicate in range(study.requested(fit, floor.REFERENCE_ARM, budget)):
            out.append((law, clusters, replicate, fit, budget, stream))
    return out


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


# --------------------------------------------------------------------------- floors, part (c)


def cell_verdicts(summary: pd.DataFrame) -> dict[tuple[str, str, int], bool]:
    """``(fit, law, J)`` to whether its positive and its control both passed."""
    passed = summary.loc[summary["property"] == FAMILY].set_index("cell")["passed"].astype(bool)
    out = {}
    for fit, law, clusters in study.grid():
        out[(fit, law, clusters)] = bool(
            passed.get(study.cell_name(fit, law, clusters, floor.REFERENCE_ARM), False)
            and passed.get(study.cell_name(fit, law, clusters, "iid_t_control"), False)
        )
    return out


def family_floors(summary: pd.DataFrame) -> dict[str, int]:
    """Each family's floor under the 5.8 rule (both mechanism configurations, every law)."""
    verdicts = cell_verdicts(summary)
    out = {}
    for family, anchor in study.ANCHORS.items():
        fits = [fit for fit, owner in study.FITS.items() if owner == family]
        passes = floor.count_passes(verdicts, fits)
        out[family] = floor.family_floor(passes, anchor=anchor)
    return out


def candidate_cells(
    rows: pd.DataFrame, family: str, *, draws: int = floor.CANDIDATE_DRAWS
) -> list[floor.CandidateCell]:
    """The part (c) readings of one family's cells on their first ``draws`` draws (M3).

    (C) at a count where R5 refuses it before any refit (no published cell) reads as withheld
    on every draw: no candidate-only coverage, every (A) coverage discordant.
    """
    family_rows = rows.loc[rows["property"] == FAMILY]
    groups = dict(tuple(family_rows.groupby("cell", sort=False)))
    out = []
    for fit, law, clusters in study.grid():
        if study.FITS[fit] != family:
            continue
        reference = groups.get(study.cell_name(fit, law, clusters, floor.REFERENCE_ARM))
        if reference is None:  # every draw of the cell was lost: no candidate reads it
            continue
        for candidate in floor.CANDIDATE_ARMS:
            name = study.cell_name(fit, law, clusters, candidate)
            frame = groups.get(name, reference.iloc[0:0])
            first = frame.loc[frame["replicate"] < draws]
            shown = first.loc[first["withheld"] == 0]
            length, interval = (1.0, None)
            if candidate != floor.REFERENCE_ARM and len(shown) >= 2:
                length, interval = floor.length_ratio(
                    frame,
                    reference,
                    draws=draws,
                    replicates=STUDY.margins.bootstrap_replicates,
                    seed=stream_seed(STUDY, "length", name),
                )
            out.append(
                floor.CandidateCell(
                    candidate=candidate,
                    configuration=fit,
                    law=law,
                    clusters=clusters,
                    pair=floor.paired_counts(frame, reference, draws=draws),
                    hits=int(shown["covered"].sum()),
                    shown=len(shown),
                    length=float(length),
                    length_high=float(interval.high) if interval is not None else 1.0,
                )
            )
    return out


def decisions(summary: pd.DataFrame, rows: pd.DataFrame) -> dict[str, floor.Decision]:
    """The part (c) decision of every compared family (plan 5.7)."""
    floors = family_floors(summary)
    out = {}
    for family in sorted(study.CANDIDATE_FAMILIES):
        fits = [fit for fit, owner in study.FITS.items() if owner == family]
        start = min(min(study.jackknife_counts(fit)) for fit in fits)
        out[family] = floor.reference_decision(
            candidate_cells(rows, family), floor_a=floors[family], jackknife_start=start
        )
    return out


def sizing(summary: pd.DataFrame) -> dict[str, floor.FamilySizing]:
    """Each family's M5 sizing off a pilot summary (``generate_property_rows(stream="pilot")``)."""
    table = summary.loc[summary["property"] == FAMILY].set_index("cell")
    out = {}
    for family, anchor in study.ANCHORS.items():
        cells = {}
        for fit, law, clusters in study.grid():
            if study.FITS[fit] != family:
                continue
            positive = study.cell_name(fit, law, clusters, floor.REFERENCE_ARM)
            control = study.cell_name(fit, law, clusters, "iid_t_control")
            if positive in table.index and control in table.index:
                cells[(fit, law, clusters)] = floor.sizing_inputs(
                    table.loc[positive], table.loc[control]
                )
        out[family] = floor.family_sizing(cells, anchor=anchor)
    return out


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
    "decisions",
    "declared_cells",
    "family_floors",
    "generate_property_rows",
    "sizing",
    "summarize_properties",
)
