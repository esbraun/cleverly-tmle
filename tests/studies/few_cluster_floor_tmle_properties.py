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
most 1.20. The rule runs with ``shown_only``: the SE ratios read the shown draws, the paired
ratio the draws both arms show, and a cell with fewer than two shown draws fails its control.
Every other arm is reported.

**Readings** (:func:`tests.studies.few_cluster_floor.floor_readings`): shown and withheld counts, the
withheld share by trigger, shown-only coverage with its exact interval, all-draw coverage with
its exact interval, the shown-only SE ratio, the over-coverage reading against 0.99, and on an
``ltmle_in_sample`` positive row the package-GLM refit's raised share with its exact interval
and the study learner's shown coverage split by whether the package GLM raised (M2, Q15).

**Part (c)** (``few_cluster_candidate``, the module's :data:`PROPERTY_RULES`, review M1): each
candidate cell on the first 2,000 draws, paired with (A)'s copy, passes when the 99% lower end of
its paired coverage gain is at least -0.03 (the harness's S7 clause: under a lost draw the
candidate reads uncovered and (A) covered). One ``few_cluster_candidate_pooled`` row per
configuration and candidate pools its pairs at or above the admissibility count, above -0.01.
Calibration and length stay reported.

**Floors and the outcome.** :func:`routes` reads each family's floor, route and red owners off
the verdicts (:func:`tests.studies.few_cluster_floor.declared_route`); :func:`decisions` reads
the part (c) decision, and the design test folds both studies' into the declared outcome.
"""

from __future__ import annotations

import functools
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

#: One key per fit of a draw: ``(law, J, replicate, fit, budget, stream)``; a (C) key's fit is
#: ``<fit>__jackknife`` (review B1).
Key = tuple[str, int, int, str, "int | None", str]


def _groups() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(study.cell_name(fit, law, clusters, arm) for arm, _ in study.arms(fit, clusters))
        for fit, law, clusters in study.grid()
    )


def _candidate_groups() -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            study.cell_name(fit, law, clusters, candidate),
            study.cell_name(fit, law, clusters, floor.reference_for(candidate)),
        )
        for fit, law, clusters in study.grid()
        for candidate in study.candidates(fit, clusters)
    )


#: The arms of one fit, size law and count, which pair each control with its positive arm;
#: each part (c) candidate with (A)'s copy; and each pooled row alone.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    FAMILY: _groups(),
    floor.CANDIDATE_FAMILY: _candidate_groups(),
    floor.POOLED_FAMILY: tuple((name,) for name in study.POOLED_CELLS),
}
#: The module's own rule for the part (c) families, and the registered ``few_cluster_reference``
#: rule in its shown-only form, so the failure accounting's worst cases read the shown draws the
#: verdict reads (``shown_only=True``).
PROPERTY_RULES = (floor.candidate_rule(), property_verdicts.few_cluster_rule(shown_only=True))


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published replication cell, with its role, size and requested replicates."""
    out = []
    for fit, law, clusters in study.grid():
        estimand = study.LTMLE_TARGET if study.FITS[fit] == "ltmle_in_sample" else study.TARGET
        published = [(FAMILY, arm, role, False) for arm, role in study.arms(fit, clusters)]
        published += [
            (floor.CANDIDATE_FAMILY, arm, role, True)
            for arm, role in study.candidate_arms(fit, clusters)
        ]
        for family, arm, role, candidate in published:
            out.append(
                PropertyCell(
                    property=family,
                    cell=study.cell_name(fit, law, clusters, arm),
                    dgp=floor.CellLaw(
                        f"{law} J{clusters}",
                        functools.partial(_cell_truths, law, estimand),
                    ),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=10 * clusters,
                    replicates=study.requested(fit, candidate=candidate),
                    seed=study.draw_seed(law, clusters, 0),
                    role=role,
                    estimand=estimand,
                )
            )
    return tuple(out)


def _cell_truths(law: str, estimand: str) -> dict[str, float]:
    """Every cell of ``law`` publishes the law's exact ATE (``law_truth``) as its truth."""
    return {estimand: study.law_truth(law)}


def _split(fit: str) -> tuple[str, bool]:
    """The fit of a key and whether it is a (C) key."""
    if fit.endswith(study.JACKKNIFE_SUFFIX):
        return fit.removesuffix(study.JACKKNIFE_SUFFIX), True
    return fit, False


def _draw(key: Key) -> pd.DataFrame:
    """The sample of a key, a function of ``(law, J, replicate, stream)`` alone."""
    law, clusters, replicate, _, _, stream = key
    return study.draw(law, clusters, study.draw_seed(law, clusters, replicate, stream))


def _cells_of(key: Key) -> list[str]:
    law, clusters, replicate, name, _, _ = key
    fit, jackknife = _split(name)
    if jackknife:
        return [
            f"{floor.CANDIDATE_FAMILY}/{study.cell_name(fit, law, clusters, floor.JACKKNIFE_ARM)}"
        ]
    out = [
        f"{FAMILY}/{study.cell_name(fit, law, clusters, arm)}"
        for arm, _ in study.arms(fit, clusters)
    ]
    if replicate < floor.CANDIDATE_DRAWS:
        out += [
            f"{floor.CANDIDATE_FAMILY}/{study.cell_name(fit, law, clusters, arm)}"
            for arm, _ in study.candidate_arms(fit, clusters)
            if arm != floor.JACKKNIFE_ARM
        ]
    return out


def _cell_of(key: Key) -> str:
    law, clusters, _, name, _, _ = key
    fit, jackknife = _split(name)
    if jackknife:
        return (
            f"{floor.CANDIDATE_FAMILY}/{study.cell_name(fit, law, clusters, floor.JACKKNIFE_ARM)}"
        )
    return f"{FAMILY}/{study.cell_name(fit, law, clusters, floor.REFERENCE_ARM)}"


def _replicate_of(key: Key) -> int:
    return key[2]


def _fit_key(key: Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    law, clusters, replicate, name, budget, stream = key
    fit, jackknife = _split(name)
    rows = study.jackknife_rows if jackknife else study.fit_rows
    return rows(fit, law, clusters, replicate, frame, budget=budget, stream=stream)


def keys(budget: int | None = None, stream: str = "property") -> list[Key]:
    """Every key of the grid, each fit's replicates capped at ``budget``: the main fit's, then
    (C)'s on the first :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS` draws."""
    out: list[Key] = []
    for fit, law, clusters in study.grid():
        for replicate in range(study.requested(fit, budget)):
            out.append((law, clusters, replicate, fit, budget, stream))
        if floor.JACKKNIFE_ARM in study.candidates(fit, clusters):
            jackknife = fit + study.JACKKNIFE_SUFFIX
            for replicate in range(study.requested(fit, budget, candidate=True)):
                out.append((law, clusters, replicate, jackknife, budget, stream))
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


def _rows_by_cell(summary: pd.DataFrame, family: str = FAMILY) -> dict[str, pd.Series]:
    table = summary.loc[summary["property"] == family]
    return {str(row["cell"]): row for _, row in table.iterrows()}


def floor_cells(
    summary: pd.DataFrame, family: str
) -> dict[tuple[str, str, int], tuple[pd.Series, pd.Series]]:
    """``(fit, law, J)`` of a floor family (or :data:`~study.D5_REPORTED`) to its positive and
    control summary rows, for every cell the summary holds."""
    rows = _rows_by_cell(summary)
    out = {}
    for fit in study.floor_fits(family):
        for law in study.laws_of(fit):
            for clusters in study.ladder(fit):
                positive = rows.get(study.cell_name(fit, law, clusters, floor.REFERENCE_ARM))
                control = rows.get(study.cell_name(fit, law, clusters, "iid_t_control"))
                if positive is not None and control is not None:
                    out[(fit, law, clusters)] = (positive, control)
    return out


def routes(summary: pd.DataFrame) -> dict[str, floor.Route]:
    """Each floor family's declared route and red owners (plan 5.8, reviews B3 and M3)."""
    return {
        family: floor.declared_route(family, floor_cells(summary, family), anchor=anchor)
        for family, anchor in study.ANCHORS.items()
    }


def reported_route(summary: pd.DataFrame) -> floor.Route:
    """The d5 configurations' own floor reading, reported evidence (review M2)."""
    return floor.declared_route(
        study.D5_REPORTED,
        floor_cells(summary, study.D5_REPORTED),
        anchor=study.ANCHORS["tmle_stacked"],
    )


def family_floors(summary: pd.DataFrame) -> dict[str, int]:
    """Each floor family's floor under the 5.8 rule (both mechanism configurations, every law)."""
    return {family: route.floor for family, route in routes(summary).items()}


def pooled_starts(summary: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Each configuration's admissibility count per candidate, off the summary's own floors.

    (B) from :math:`F_f(A)`, (C) from :math:`\\max(F_f(A), 2V)`, the configuration's first
    jackknife count (plan 5.7 rule 1). A family whose cells the summary does not all hold (a
    differential subset) reads the floor of the counts it holds, or its anchor.
    """
    out = {}
    for family, anchor in study.ANCHORS.items():
        if family not in study.CANDIDATE_FAMILIES:
            continue
        cells = floor_cells(summary, family)
        verdicts = {
            key: bool(positive["passed"]) and bool(control["passed"])
            for key, (positive, control) in cells.items()
        }
        try:
            floor_a = floor.family_floor(
                floor.count_passes(verdicts, study.floor_fits(family)), anchor=anchor
            )
        except ValueError:
            floor_a = anchor
        for fit in study.configurations():
            if study.FITS[fit] != family:
                continue
            out[(fit, floor.WANG_ARM)] = floor_a
            out[(fit, floor.JACKKNIFE_ARM)] = max(floor_a, min(study.jackknife_counts(fit)))
    return out


def pooled_verdicts(summary: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """Each pooled row's verdict, by ``(configuration, candidate)``."""
    table = summary.loc[summary["property"] == floor.POOLED_FAMILY]
    return {
        tuple(str(row["cell"]).split("__")): bool(row["property_passed"])  # type: ignore[misc]
        for _, row in table.iterrows()
    }


def candidate_cells(
    summary: pd.DataFrame, rows: pd.DataFrame, family: str, *, draws: int = floor.CANDIDATE_DRAWS
) -> list[floor.CandidateCell]:
    """The part (c) readings of one family's cells on their first ``draws`` draws.

    (A) reads its own ``t_reference`` rows; (B) and (C) read their ``few_cluster_candidate``
    cells against (A)'s copy, with the verdict row's S7 lower end (clause (i)). (C) at a count
    where R5 refuses it before any refit has no cell, and no admissibility count reaches it.
    """
    reference_rows = rows.loc[rows["property"] == FAMILY]
    candidate_rows = rows.loc[rows["property"] == floor.CANDIDATE_FAMILY]
    references = dict(tuple(reference_rows.groupby("cell", sort=False)))
    pairs = dict(tuple(candidate_rows.groupby("cell", sort=False)))
    verdicts = _rows_by_cell(summary, floor.CANDIDATE_FAMILY)
    out = []
    for fit, law, clusters in study.grid():
        if study.FITS[fit] != family:
            continue
        reference = references.get(study.cell_name(fit, law, clusters, floor.REFERENCE_ARM))
        if reference is not None:
            first = reference.loc[reference["replicate"] < draws]
            shown = first.loc[first["withheld"] == 0]
            out.append(
                floor.CandidateCell(
                    candidate=floor.REFERENCE_ARM,
                    configuration=fit,
                    law=law,
                    clusters=clusters,
                    pair=floor.paired_counts(reference, reference, draws=draws),
                    hits=int(shown["covered"].sum()),
                    shown=len(shown),
                )
            )
        for candidate in study.candidates(fit, clusters):
            name = study.cell_name(fit, law, clusters, candidate)
            control = pairs.get(study.cell_name(fit, law, clusters, floor.reference_for(candidate)))
            frame = pairs.get(name)
            if control is None:  # every draw of the cell was lost: (A) has no copy to pair
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
                    configuration=fit,
                    law=law,
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


def decisions(summary: pd.DataFrame, rows: pd.DataFrame) -> dict[str, floor.Decision]:
    """The part (c) decision of every compared family (plan 5.7)."""
    floors = family_floors(summary)
    pooled = pooled_verdicts(summary)
    out = {}
    for family in sorted(study.CANDIDATE_FAMILIES):
        fits = [fit for fit in study.configurations() if study.FITS[fit] == family]
        start = min(min(study.jackknife_counts(fit)) for fit in fits)
        out[family] = floor.reference_decision(
            candidate_cells(summary, rows, family),
            floor_a=floors[family],
            jackknife_start=start,
            pooled=pooled,
        )
    return out


def sizing(summary: pd.DataFrame) -> dict[str, floor.FamilySizing]:
    """Each floor family's M5 sizing off a pilot summary (``generate_property_rows(stream="pilot")``)."""
    out = {}
    for family, anchor in study.ANCHORS.items():
        cells = {
            key: floor.sizing_inputs(positive, control)
            for key, (positive, control) in floor_cells(summary, family).items()
        }
        out[family] = floor.family_sizing(cells, anchor=anchor)
    return out


def synthetic_columns(cell: Any, rng: Any) -> dict[str, Any]:
    """The extra columns of a synthetic row: a shown interval, no package-GLM refit."""
    del rng
    clusters = cell.n // 10
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
#: cell, so a subset with one positive-and-control group per clause and threshold binds every
#: clause; ``docs/development/method-benchmarking.md`` records the measured time.
DIFFERENTIAL_BUDGET: Final[int] = 200
#: The groups the differential reads, one per family and construction, each at one count
#: (4, 10, 20 or 39 across the groups), with its part (c) pairs and the pooled rows of their
#: configurations: every arm kind, the fold-evaluated :math:`t_{J-2}` arm, (B) at
#: :math:`p = 5`, and a pooled row. The pooled functions are shared with S-b, whose subset
#: gives a pooled row three pairs.
DIFFERENTIAL_GROUPS: Final[tuple[tuple[str, str, int], ...]] = (
    ("tmle_in_sample_known", "equal10", 4),
    ("tmle_stacked_estimated", "unequal_informative", 10),
    ("tmle_fold_evaluated_known", "equal10", 10),
    ("drtmle_known", "unequal_informative", 39),
    ("tmle_stacked_d5_estimated", "unequal_informative", 10),
    ("ltmle_in_sample", "unequal_informative", 20),
)


def differential_cells() -> frozenset[str]:
    """The ``family/cell`` names of :data:`DIFFERENTIAL_GROUPS`."""
    out = set()
    for fit, law, clusters in DIFFERENTIAL_GROUPS:
        out |= {
            f"{FAMILY}/{study.cell_name(fit, law, clusters, arm)}"
            for arm, _ in study.arms(fit, clusters)
        }
        out |= {
            f"{floor.CANDIDATE_FAMILY}/{study.cell_name(fit, law, clusters, arm)}"
            for arm, _ in study.candidate_arms(fit, clusters)
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
    "decisions",
    "declared_cells",
    "family_floors",
    "generate_property_rows",
    "routes",
    "sizing",
    "summarize_properties",
)
