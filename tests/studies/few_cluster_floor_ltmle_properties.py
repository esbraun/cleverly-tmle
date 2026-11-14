r"""The ``few_cluster_reference`` family of ``few-cluster-floor-cross-fitted-ltmle``.

Each key fits one draw of a ``(law, J)`` cell
(``stream_seed(STUDY, "property", f"{law}/J{J}", replicate)``; the pilot reads ``"pilot"``) and
writes one row per arm (:func:`tests.studies.few_cluster_floor_ltmle.fit_rows`). A fit that
raises is a lost key, a miss under the failure accounting.

The gated rule is the registered two-clause rule of ``clustered-few-cluster-tmle``
(:func:`~tests.studies.evidence.property_verdicts.few_cluster_reference_verdicts`): G1 and G2 on
``t_reference``, both control clauses on ``iid_t_control``. The readings are
:func:`tests.studies.few_cluster_floor.floor_readings`. Part (c) is the module's own
:data:`PROPERTY_RULES` (review M1): each ``few_cluster_candidate`` pair passes when the 99% lower
end of the candidate's paired coverage gain over (A) is at least -0.03 (S7), and each
``few_cluster_candidate_pooled`` row when its pooled gain lies above -0.01. :func:`route` reads
the family's floor, route and red owners, and :func:`decision` the part (c) decision.
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

#: One key per fit of a draw: ``(law, J, replicate, fit, budget, stream)``, the fit
#: :data:`~tests.studies.few_cluster_floor_ltmle.FIT` or its (C) key ``JACKKNIFE_FIT``.
Key = tuple[str, int, int, str, "int | None", str]

#: The arms of one size law and count, which pair the control with its positive arm; each
#: part (c) candidate with (A)'s copy; and each pooled row alone.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    FAMILY: tuple(
        tuple(study.cell_name(sizes, clusters, arm) for arm, _ in study.arms(clusters))
        for sizes, clusters in study.grid()
    ),
    floor.CANDIDATE_FAMILY: tuple(
        (
            study.cell_name(sizes, clusters, candidate),
            study.cell_name(sizes, clusters, floor.reference_for(candidate)),
        )
        for sizes, clusters in study.grid()
        for candidate in study.candidates(clusters)
    ),
    floor.POOLED_FAMILY: tuple((name,) for name in study.POOLED_CELLS),
}
#: The module's own rule for the part (c) families.
PROPERTY_RULES = (floor.candidate_rule(),)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published replication cell, with its role, size and requested replicates."""
    published = [
        (FAMILY, sizes, clusters, arm, role, False)
        for sizes, clusters in study.grid()
        for arm, role in study.arms(clusters)
    ] + [
        (floor.CANDIDATE_FAMILY, sizes, clusters, arm, role, True)
        for sizes, clusters in study.grid()
        for arm, role in study.candidate_arms(clusters)
    ]
    return tuple(
        PropertyCell(
            property=family,
            cell=study.cell_name(sizes, clusters, arm),
            dgp=(sizes, clusters),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=study.CLUSTER_SIZE * clusters,
            replicates=study.requested(candidate=candidate),
            seed=study.draw_seed(sizes, clusters, 0),
            role=role,
            estimand=study.TARGET,
        )
        for family, sizes, clusters, arm, role, candidate in published
    )


def _draw(key: Key) -> tuple[pd.DataFrame, dict[str, float]]:
    sizes, clusters, replicate, _, _, stream = key
    return study.draw(sizes, clusters, study.draw_seed(sizes, clusters, replicate, stream))


def _cells_of(key: Key) -> list[str]:
    sizes, clusters, replicate, fit, _, _ = key
    if fit == study.JACKKNIFE_FIT:
        return [f"{floor.CANDIDATE_FAMILY}/{study.cell_name(sizes, clusters, floor.JACKKNIFE_ARM)}"]
    out = [f"{FAMILY}/{study.cell_name(sizes, clusters, arm)}" for arm, _ in study.arms(clusters)]
    if replicate < floor.CANDIDATE_DRAWS:
        out += [
            f"{floor.CANDIDATE_FAMILY}/{study.cell_name(sizes, clusters, arm)}"
            for arm, _ in study.candidate_arms(clusters)
            if arm != floor.JACKKNIFE_ARM
        ]
    return out


def _cell_of(key: Key) -> str:
    sizes, clusters, _, fit, _, _ = key
    if fit == study.JACKKNIFE_FIT:
        return f"{floor.CANDIDATE_FAMILY}/{study.cell_name(sizes, clusters, floor.JACKKNIFE_ARM)}"
    return f"{FAMILY}/{study.cell_name(sizes, clusters, floor.REFERENCE_ARM)}"


def _replicate_of(key: Key) -> int:
    return key[2]


def _fit_key(key: Key, drawn: tuple[pd.DataFrame, dict[str, float]]) -> list[dict[str, Any]]:
    sizes, clusters, replicate, fit, budget, _ = key
    rows = study.jackknife_rows if fit == study.JACKKNIFE_FIT else study.fit_rows
    return rows(sizes, clusters, replicate, drawn, budget=budget)


def keys(budget: int | None = None, stream: str = "property") -> list[Key]:
    """Every key of the grid, each cell's replicates capped at ``budget``: the main fit's, then
    (C)'s on the first :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS` draws."""
    out: list[Key] = []
    for sizes, clusters in study.grid():
        for replicate in range(study.requested(budget)):
            out.append((sizes, clusters, replicate, study.FIT, budget, stream))
        if floor.JACKKNIFE_ARM in study.candidates(clusters):
            for replicate in range(study.requested(budget, candidate=True)):
                out.append((sizes, clusters, replicate, study.JACKKNIFE_FIT, budget, stream))
    return out


def generate_property_rows(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    stream: str = "property",
    timed: bool = False,
    record_every: bool = False,
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pilot or a smoke.

    ``timed`` (the pilot's) keeps each key's CPU seconds on its rows
    (:data:`~tests.studies.few_cluster_floor.TIMING_COLUMNS`), which a run does not publish.
    ``record_every`` (the pilot's) records every exception of a key as a lost key
    (:func:`~tests.studies.few_cluster_floor.recording_every_error`).
    """
    fit_one = floor.timed(_fit_key) if timed else _fit_key
    if record_every:
        fit_one = floor.recording_every_error(fit_one)
    outcomes = fit_properties(
        keys(budget, stream),
        fit_one,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    timing = floor.TIMING_COLUMNS if timed else ()
    return rows.reindex(
        columns=[*REPLICATE_COLUMNS, *floor.EXTRA_COLUMNS, *floor.PACKAGE_COLUMNS, *timing]
    ).sort_values(["property", "cell", "replicate"], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the registered few-cluster rule, the floor readings, and part (c)."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(*FEW_CLUSTER_COLUMNS, *floor.READING_COLUMNS, *floor.CANDIDATE_COLUMNS),
        rate_labels=(),
    )
    few_cluster_reference_verdicts(summary, rows, STUDY, PROPERTY_GROUPS[FAMILY], shown_only=True)
    floor.floor_readings(summary, rows, STUDY)
    floor.candidate_verdicts(summary, rows, STUDY, PROPERTY_GROUPS[floor.CANDIDATE_FAMILY])
    rates.extend(floor.pooled_rows(summary, rows, STUDY, pooled_starts(summary)))
    return finish(summary, rates)


# --------------------------------------------------------------------------- floors, part (c)


def floor_cells(summary: pd.DataFrame) -> dict[tuple[str, str, int], tuple[pd.Series, pd.Series]]:
    """``(fit, law, J)`` to its positive and control summary rows, for every cell held."""
    table = summary.loc[summary["property"] == FAMILY]
    rows = {str(row["cell"]): row for _, row in table.iterrows()}
    out = {}
    for sizes, clusters in study.grid():
        positive = rows.get(study.cell_name(sizes, clusters, floor.REFERENCE_ARM))
        control = rows.get(study.cell_name(sizes, clusters, "iid_t_control"))
        if positive is not None and control is not None:
            out[(study.FIT, sizes, clusters)] = (positive, control)
    return out


def route(summary: pd.DataFrame) -> floor.Route:
    """The family's declared route and red owners (plan 5.8, reviews B3 and M3)."""
    return floor.declared_route(study.FIT, floor_cells(summary), anchor=study.ANCHOR)


def family_floor(summary: pd.DataFrame) -> int:
    """The family's floor under the 5.8 rule: every count passes on both size laws."""
    return route(summary).floor


def pooled_starts(summary: pd.DataFrame) -> dict[tuple[str, str], int]:
    """The admissibility count per candidate, off the summary's own floor (plan 5.7 rule 1).

    A summary that does not hold every count (a differential subset) reads the floor of the
    counts it holds, or the anchor.
    """
    cells = floor_cells(summary)
    verdicts = {
        key: bool(positive["passed"]) and bool(control["passed"])
        for key, (positive, control) in cells.items()
    }
    try:
        floor_a = floor.family_floor(floor.count_passes(verdicts, [study.FIT]), anchor=study.ANCHOR)
    except ValueError:
        floor_a = study.ANCHOR
    return {
        (study.FIT, floor.WANG_ARM): floor_a,
        (study.FIT, floor.JACKKNIFE_ARM): max(floor_a, min(study.jackknife_counts())),
    }


def candidate_cells(
    summary: pd.DataFrame, rows: pd.DataFrame, *, draws: int = floor.CANDIDATE_DRAWS
) -> list[floor.CandidateCell]:
    """The part (c) readings of every cell on its first ``draws`` draws (M1, M3)."""
    reference_rows = rows.loc[rows["property"] == FAMILY]
    candidate_rows = rows.loc[rows["property"] == floor.CANDIDATE_FAMILY]
    references = dict(tuple(reference_rows.groupby("cell", sort=False)))
    pairs = dict(tuple(candidate_rows.groupby("cell", sort=False)))
    table = summary.loc[summary["property"] == floor.CANDIDATE_FAMILY]
    verdicts = {str(row["cell"]): row for _, row in table.iterrows()}
    out = []
    for sizes, clusters in study.grid():
        reference = references.get(study.cell_name(sizes, clusters, floor.REFERENCE_ARM))
        if reference is not None:
            first = reference.loc[reference["replicate"] < draws]
            shown = first.loc[first["withheld"] == 0]
            out.append(
                floor.CandidateCell(
                    candidate=floor.REFERENCE_ARM,
                    configuration=study.FIT,
                    law=sizes,
                    clusters=clusters,
                    pair=floor.paired_counts(reference, reference, draws=draws),
                    hits=int(shown["covered"].sum()),
                    shown=len(shown),
                )
            )
        for candidate in study.candidates(clusters):
            name = study.cell_name(sizes, clusters, candidate)
            control = pairs.get(study.cell_name(sizes, clusters, floor.reference_for(candidate)))
            frame = pairs.get(name)
            if control is None:
                control = frame.iloc[0:0] if frame is not None else candidate_rows.iloc[0:0]
            if frame is None:
                frame = control.iloc[0:0]
            shown = frame.loc[(frame["replicate"] < draws) & (frame["withheld"] == 0)]
            length, high = float("nan"), float("nan")
            if len(shown) >= 2 and len(control):
                length, interval = floor.length_ratio(
                    frame,
                    control,
                    draws=draws,
                    replicates=STUDY.margins.bootstrap_replicates,
                    seed=stream_seed(STUDY, "length", name),
                )
                high = interval.high
            verdict = verdicts.get(name)
            out.append(
                floor.CandidateCell(
                    candidate=candidate,
                    configuration=study.FIT,
                    law=sizes,
                    clusters=clusters,
                    pair=floor.paired_counts(frame, control, draws=draws),
                    hits=int(shown["covered"].sum()),
                    shown=len(shown),
                    length=float(length),
                    length_high=float(high),
                    gain_low=float(verdict["coverage_gain_ci_lower"])
                    if verdict is not None
                    else float("nan"),
                )
            )
    return out


def decision(summary: pd.DataFrame, rows: pd.DataFrame) -> floor.Decision:
    """The part (c) decision of the family (plan 5.7)."""
    table = summary.loc[summary["property"] == floor.POOLED_FAMILY]
    pooled = {
        tuple(str(row["cell"]).split("__")): bool(row["property_passed"])
        for _, row in table.iterrows()
    }
    return floor.reference_decision(
        candidate_cells(summary, rows),
        floor_a=family_floor(summary),
        jackknife_start=min(study.jackknife_counts()),
        pooled=pooled,  # type: ignore[arg-type]
    )


def sizing(summary: pd.DataFrame) -> floor.FamilySizing:
    """The family's M5 sizing off a pilot summary (``generate_property_rows(stream="pilot")``)."""
    cells = {
        key: floor.sizing_inputs(positive, control)
        for key, (positive, control) in floor_cells(summary).items()
    }
    return floor.family_sizing(cells, anchor=study.ANCHOR)


def synthetic_columns(cell: Any, rng: Any) -> dict[str, Any]:
    """The extra columns of a synthetic row: a shown interval, the package GLM ran."""
    del rng
    _, clusters = cell.dgp
    return {
        "withheld": 0,
        "trigger": "",
        "reference_df": float(clusters - 2),
        "clusters": clusters,
        "half_width": float("nan"),
        "failed_refits": 0,
        "jackknife_error": "",
    }


#: The property differential's declared size (review B4): each cell capped at this many
#: replicates, on :data:`DIFFERENTIAL_GROUPS` only. The rule is one shared rule over every
#: cell, so a subset with a positive-and-control group and a part (c) pair binds every clause.
DIFFERENTIAL_BUDGET: Final[int] = 200
#: The groups the differential reads: the smallest count, the first count with a (C) pair,
#: the anchor and 39, alternating the size laws.
DIFFERENTIAL_GROUPS: Final[tuple[tuple[str, int], ...]] = (
    ("equal40", 4),
    ("unequal40", 10),
    ("equal40", 20),
    ("unequal40", 39),
)


def differential_cells() -> frozenset[str]:
    """The ``family/cell`` names of :data:`DIFFERENTIAL_GROUPS`."""
    out = set()
    for sizes, clusters in DIFFERENTIAL_GROUPS:
        out |= {
            f"{FAMILY}/{study.cell_name(sizes, clusters, arm)}" for arm, _ in study.arms(clusters)
        }
        out |= {
            f"{floor.CANDIDATE_FAMILY}/{study.cell_name(sizes, clusters, arm)}"
            for arm, _ in study.candidate_arms(clusters)
        }
    return frozenset(out)


#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS
#: The property failure accounting's bound (:data:`tests.studies.few_cluster_floor.FAILURE_BOUND`);
#: the primary's is :data:`tests.studies.few_cluster_floor.PRIMARY_FAILURE_BOUND`.
FAILURE_BOUND = floor.FAILURE_BOUND


__all__: Sequence[str] = (
    "PROPERTY_GROUPS",
    "PROPERTY_RULES",
    "candidate_cells",
    "decision",
    "declared_cells",
    "family_floor",
    "generate_property_rows",
    "route",
    "sizing",
    "summarize_properties",
)
