r"""The ``few_cluster_reference`` family of ``few-cluster-arm-means-cross-fitted-ltmle`` (S-k2).

Each key fits the ``few-cluster-floor-cross-fitted-ltmle`` subject on one draw of a
``(law, J)`` cell, ``stream_seed(STUDY, "property", f"{law}/J{J}", replicate)`` (the pilot reads
``"pilot"``), and writes the arms of its three estimands
(:func:`tests.studies.few_cluster_arm_means_ltmle.fit_rows`). A fit that raises is a lost key.

**Gated** (``few_cluster_rule(shown_only=True)``): G1 and G2 on each regimen mean's
``t_reference`` row and both clauses on its ``iid_t_control`` row, on both laws. The contrast's
rows are reported. :func:`route` reads the regimen means' floor with
:func:`~tests.studies.few_cluster_floor.arm_mean_floor` from the contrast floor 4, not capped at
the default 20. The pilot's sizing (:func:`pilot_sizing_table`) is the rule of
:mod:`tests.studies.few_cluster_arm_means_properties`.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, Final

import pandas as pd

from cleverly._inference_status import CLUSTER_INTERVAL_FLOORS
from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_arm_means_ltmle as study
from tests.studies import few_cluster_arm_means_properties as point
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_pilot as pilot
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import (
    FEW_CLUSTER_COLUMNS,
    apply_shared_verdicts,
    few_cluster_reference_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties

STUDY = study.STUDY
FAMILY: Final[str] = study.FAMILY

#: One key per draw: ``(law, J, replicate, budget, stream)``.
Key = tuple[str, int, int, "int | None", str]

#: The arms of one law, count and estimand.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    FAMILY: tuple(
        tuple(study.cell_name(estimand, sizes, clusters, arm) for arm, _ in study.arms(estimand))
        for sizes, clusters in study.grid()
        for estimand in study.ESTIMANDS
    )
}
#: The registered rule in its shown-only form.
PROPERTY_RULES = (property_verdicts.few_cluster_rule(shown_only=True),)


def _cell_truths(sizes: str, clusters: int) -> dict[str, float]:
    """The law's exact truths, which a draw of the cell returns unchanged."""
    _, truths = study.draw(sizes, clusters, study.draw_seed(sizes, clusters, 0))
    return {name: float(truths[name]) for name in study.ESTIMANDS}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published replication cell, with its role, size and requested replicates."""
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=study.cell_name(estimand, sizes, clusters, arm),
            dgp=floor.CellLaw(
                f"{sizes} J{clusters}", functools.partial(_cell_truths, sizes, clusters)
            ),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=study.CLUSTER_SIZE * clusters,
            replicates=study.requested(),
            seed=study.draw_seed(sizes, clusters, 0),
            role=role,
            estimand=estimand,
        )
        for sizes, clusters in study.grid()
        for estimand in study.ESTIMANDS
        for arm, role in study.arms(estimand)
    )


def _draw(key: Key) -> tuple[pd.DataFrame, dict[str, float]]:
    sizes, clusters, replicate, _, stream = key
    return study.draw(sizes, clusters, study.draw_seed(sizes, clusters, replicate, stream))


def _cells_of(key: Key) -> list[str]:
    sizes, clusters, _, _, _ = key
    return [
        f"{FAMILY}/{study.cell_name(estimand, sizes, clusters, arm)}"
        for estimand in study.ESTIMANDS
        for arm, _ in study.arms(estimand)
    ]


def _cell_of(key: Key) -> str:
    sizes, clusters, _, _, _ = key
    return f"{FAMILY}/{study.cell_name(study.ESTIMANDS[0], sizes, clusters, floor.REFERENCE_ARM)}"


def _replicate_of(key: Key) -> int:
    return key[2]


def _fit_key(key: Key, drawn: tuple[pd.DataFrame, dict[str, float]]) -> list[dict[str, Any]]:
    sizes, clusters, replicate, budget, _ = key
    return study.fit_rows(sizes, clusters, replicate, drawn, budget=budget)


def keys(budget: int | None = None, stream: str = "property") -> list[Key]:
    """Every key of the grid."""
    return [
        (sizes, clusters, replicate, budget, stream)
        for sizes, clusters in study.grid()
        for replicate in range(study.requested(budget))
    ]


def generate_property_rows(
    *,
    n_jobs: int = STUDY_JOBS,
    budget: int | None = None,
    stream: str = "property",
    timed: bool = False,
    record_every: bool = False,
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a pilot or a smoke.

    ``timed`` and ``record_every`` are the pilot's (:mod:`tests.studies.few_cluster_floor_pilot`).
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
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *floor.EXTRA_COLUMNS, *timing]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the registered few-cluster rule and the floor readings."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(*FEW_CLUSTER_COLUMNS, *floor.READING_COLUMNS),
        rate_labels=(),
    )
    few_cluster_reference_verdicts(summary, rows, STUDY, PROPERTY_GROUPS[FAMILY], shown_only=True)
    floor.floor_readings(summary, rows, STUDY)
    return finish(summary, rates)


# --------------------------------------------------------------------------------- the floor


def floor_cells(
    summary: pd.DataFrame,
) -> dict[tuple[str, str, str, int], tuple[pd.Series, pd.Series]]:
    """``(configuration, law, estimand, J)`` of each gated regimen mean the summary holds, to its
    positive and control summary rows."""
    table = summary.loc[summary["property"] == FAMILY]
    rows = {str(row["cell"]): row for _, row in table.iterrows()}
    out = {}
    for sizes, clusters in study.grid():
        for estimand in study.ARM_MEANS:
            positive = rows.get(study.cell_name(estimand, sizes, clusters, floor.REFERENCE_ARM))
            control = rows.get(study.cell_name(estimand, sizes, clusters, "iid_t_control"))
            if positive is not None and control is not None:
                out[(study.FIT, sizes, estimand, clusters)] = (positive, control)
    return out


def contrast_floor() -> int:
    """The shipped contrast floor of cross-fitted ``LTMLE``."""
    return int(CLUSTER_INTERVAL_FLOORS[study.FLOOR_FAMILY])


def route(summary: pd.DataFrame) -> floor.Route:
    """The regimen means' floor :math:`F_m`, its route and its red owners."""
    return floor.arm_mean_floor(
        floor_cells(summary),
        family=study.FLOOR_FAMILY,
        anchor=study.ANCHOR,
        contrast_floor=contrast_floor(),
        ladder=study.LADDER,
        owner=study.RED_OWNER,
        prefix=FAMILY,
        clause_one_laws=study.CLAUSE_ONE_LAWS,
    )


def routes(summary: pd.DataFrame) -> dict[str, floor.Route]:
    """:func:`route` keyed by the floor family."""
    return {study.FLOOR_FAMILY: route(summary)}


# ------------------------------------------------------------------------------------- pilot


def family_sizing(summary: pd.DataFrame) -> dict[str, Any]:
    """The pilot sizing (M5 over the gated cells at or above :math:`\\hat F_m`, or at the top count), predicted
    floor and predicted reds, by the rule of the point study."""
    contrast = contrast_floor()
    cells = floor_cells(summary)
    inputs = {
        key: floor.sizing_inputs(positive, control) for key, (positive, control) in cells.items()
    }
    coverage: dict[int, list[float]] = {}
    for key, values in inputs.items():
        coverage.setdefault(key[3], []).append(values["coverage"])
    start = point.expected_floor(coverage, ladder=study.LADDER, contrast=contrast)
    # With no count expected to pass, the top count still decides between it and 40.
    sized = min(start, max(study.LADDER))
    above = [values for key, values in inputs.items() if key[3] >= sized]
    replicates = floor.required_replicates(above)
    plugin = floor.family_pass_probability(above, replicates)
    predictive = 1.0
    for values in above:
        predictive *= floor.predictive_g1(
            round(values["hits"]), int(values["pilot_draws"]), replicates
        )
    red_counts: set[int] = set()
    reds: dict[str, str] = {}
    for key, (positive, control) in sorted(cells.items()):
        values = inputs[key]
        if point.positive_pass(values, replicates) < pilot.RED_PROBABILITY:
            red_counts.add(key[3])
            reds[f"{FAMILY}/{positive['cell']}"] = floor.red_reading(positive)
        if pilot.control_power(values, replicates) < pilot.CONTROL_POWER:
            red_counts.add(key[3])
            reds[f"{FAMILY}/{control['cell']}"] = floor.CONTROL_READING
    return {
        "expected_floor": start,
        "predicted_floor": point.predicted_floor(
            red_counts, ladder=study.LADDER, contrast=contrast
        ),
        "replicates": replicates,
        "plugin": plugin,
        "predictive": predictive,
        "target_met": plugin >= floor.FAMILY_PASS_TARGET,
        "reds": reds,
    }


def primary_pass(summary: pd.DataFrame) -> float:
    """The primary cell's plug-in G1 pass probability at 1,000 replications off the pilot."""
    table = summary.loc[summary["property"] == FAMILY].set_index("cell")
    name = study.cell_name(
        study.PRIMARY_ESTIMAND, study.PRIMARY_LAW, study.PRIMARY_CLUSTERS, floor.REFERENCE_ARM
    )
    coverage = float(table.loc[name, "all_draw_coverage"])
    return floor.g1_pass_probability(coverage, study.PRIMARY_REPLICATES)


def pilot_sizing_table(summary: pd.DataFrame) -> pd.DataFrame:
    """``pilot-sizing.csv``: the sizing and predicted floor, each predicted red, the primary."""
    result = family_sizing(summary)
    out: list[dict[str, Any]] = [
        {
            "family": study.FLOOR_FAMILY,
            "kind": "family",
            "cell": "",
            "reading": "",
            **{key: value for key, value in result.items() if key != "reds"},
        }
    ]
    out += [
        {"family": study.FLOOR_FAMILY, "kind": "predicted red", "cell": cell, "reading": reading}
        for cell, reading in result["reds"].items()
    ]
    out.append(
        {
            "family": study.FLOOR_FAMILY,
            "kind": "primary",
            "cell": study.cell_name(
                study.PRIMARY_ESTIMAND,
                study.PRIMARY_LAW,
                study.PRIMARY_CLUSTERS,
                floor.REFERENCE_ARM,
            ),
            "reading": "",
            "replicates": study.PRIMARY_REPLICATES,
            "g1_pass": primary_pass(summary),
        }
    )
    return pd.DataFrame(out)


def family_of(fit: str) -> str:
    """The floor family of every fit token, for the pilot's timing table."""
    del fit
    return study.FLOOR_FAMILY


def timed_cell(cell: str) -> bool:
    """Whether a row carries its key's timing once: the first estimand's ``t_reference``."""
    estimand, _, _, arm = study.parse(cell)
    return arm == floor.REFERENCE_ARM and estimand == study.ESTIMANDS[0]


def synthetic_columns(cell: Any, rng: Any) -> dict[str, Any]:
    """The extra columns of a synthetic row: a shown interval at ``J - 2`` degrees of freedom."""
    del rng
    clusters = cell.n // study.CLUSTER_SIZE
    return {
        "withheld": 0,
        "trigger": "",
        "reference_df": float(clusters - 2),
        "clusters": clusters,
        "half_width": float("nan"),
        "failed_refits": 0,
        "jackknife_error": "",
    }


#: The property differential's declared size: each cell capped at this many replicates, on
#: :data:`DIFFERENTIAL_GROUPS` only, every estimand of each: one count per law.
DIFFERENTIAL_BUDGET: Final[int] = 200
DIFFERENTIAL_GROUPS: Final[tuple[tuple[str, int], ...]] = (("equal40", 4), ("unequal40", 20))


def differential_cells() -> frozenset[str]:
    """The ``family/cell`` names of :data:`DIFFERENTIAL_GROUPS`."""
    return frozenset(
        f"{FAMILY}/{study.cell_name(estimand, sizes, clusters, arm)}"
        for sizes, clusters in DIFFERENTIAL_GROUPS
        for estimand in study.ESTIMANDS
        for arm, _ in study.arms(estimand)
    )


#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS
#: The property failure accounting's bound (:data:`tests.studies.few_cluster_floor.FAILURE_BOUND`).
FAILURE_BOUND = floor.FAILURE_BOUND


__all__: Sequence[str] = (
    "PROPERTY_GROUPS",
    "PROPERTY_RULES",
    "declared_cells",
    "generate_property_rows",
    "pilot_sizing_table",
    "route",
    "summarize_properties",
)
