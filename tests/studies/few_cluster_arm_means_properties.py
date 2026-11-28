r"""The ``few_cluster_reference`` family of ``few-cluster-arm-means`` (X54 part (k)).

Each key fits one configuration of :data:`tests.studies.few_cluster_arm_means.FITS` on one draw
of a ``(law, J)`` cell, ``stream_seed(STUDY, "property", f"{law}/J{J}", replicate)`` (the pilot
reads ``"pilot"``), and writes one row per arm of each of its three estimands
(:func:`tests.studies.few_cluster_arm_means.fit_rows`). Every configuration of a
``(law, J, replicate)`` reads the same sample. A fit that raises is a lost key: a miss in every
cell of the fit at that draw.

**Gated** (the registered two-clause rule, ``few_cluster_rule(shown_only=True)``): G1 and G2 on
the ``t_reference`` row of each arm mean on all four laws, and both clauses on its
``iid_t_control`` row. The summary's ``passed`` is the registered rule on every law; clause 1
leaves the verdict on the arm-mean laws only (:func:`routes`, ``clause_one_laws``), and a control
whose only failing condition is clause 1 on an ATE law reads ``control unpowered by law``. The
contrast is gated on the ATE laws for the estimated configurations at ``"auto"``, the transfer
check of the shipped contrast floors (:func:`transfer_owners`); every other contrast row is
reported.

**Floor.** :func:`routes` reads each family's arm-mean floor :math:`F_m` with
:func:`~tests.studies.few_cluster_floor.arm_mean_floor`, from the family's contrast floor
(:func:`contrast_floor`) and not capped at the default, so a red at or above the default raises
:math:`F_m` above it.

**Pilot** (:func:`pilot_sizing_table`). Each family's R is the M5 plug-in rule (from 4,000 in
steps of 400 to 6,000, family-wise target 0.95) over its gated cells at or above its expected
arm-mean floor (:func:`expected_floor`, not capped at the anchor), clause 1 counted on the
arm-mean laws only. The table lists the predicted floor (:func:`predicted_floor`), each
predicted red with its reading, the primary rule's pick (:func:`primary_pick`) and the known
hard cell's G1 and G2 pass probabilities (:func:`hard_cell_prediction`).
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import pandas as pd

from cleverly._inference_status import CLUSTER_INTERVAL_FLOORS, FEW_CLUSTER_THRESHOLD
from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_arm_means as study
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
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.replication import fit_properties

STUDY = study.STUDY
FAMILY: Final[str] = study.FAMILY

#: One key per fit of a draw: ``(law, J, replicate, fit, budget, stream)``.
Key = tuple[str, int, int, str, "int | None", str]
#: The reading of a predicted red that no declared reading holds.
DEFECT_SUSPECT: Final[str] = "defect suspect"
#: The contrast floor ``arm_mean_floor`` reads for in-sample ``LTMLE`` when the known hard cell
#: does not pass or does not agree with S-a's primary (plan 5.5, last item).
HARD_CELL_CONTRAST_FLOOR: Final[int] = 15
#: S-a's committed primary, whose row at the hard cell the property cell is compared with.
SOURCE_PERFORMANCE = (
    ROOT / "tests" / "canonical" / "few_cluster_floor_tmle" / "performance-tests.csv"
)
SOURCE_PRIMARY: Final[str] = "floor_unequal_informative_j10"


def _groups() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            study.cell_name(fit, estimand, law, clusters, arm)
            for arm, _ in study.arms(fit, law, clusters, estimand)
        )
        for fit, law, clusters in study.grid()
        for estimand in study.estimands_of(fit)
    )


#: The arms of one fit, law, count and estimand.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {FAMILY: _groups()}
#: The registered rule in its shown-only form.
PROPERTY_RULES = (property_verdicts.few_cluster_rule(shown_only=True),)


def _cell_truths(law: str) -> dict[str, float]:
    return study.law_truth(law)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published replication cell, with its role, size and requested replicates."""
    out = []
    for fit, law, clusters in study.grid():
        for estimand in study.estimands_of(fit):
            for arm, role in study.arms(fit, law, clusters, estimand):
                out.append(
                    PropertyCell(
                        property=FAMILY,
                        cell=study.cell_name(fit, estimand, law, clusters, arm),
                        dgp=floor.CellLaw(
                            f"{law} J{clusters}", functools.partial(_cell_truths, law)
                        ),
                        outcome_learner=lambda: None,
                        treatment_learner=lambda: None,
                        n=10 * clusters,
                        replicates=study.requested(fit),
                        seed=study.draw_seed(law, clusters, 0),
                        role=role,
                        estimand=estimand,
                    )
                )
    return tuple(out)


def _draw(key: Key) -> pd.DataFrame:
    law, clusters, replicate, _, _, stream = key
    return study.draw(law, clusters, study.draw_seed(law, clusters, replicate, stream))


def _cells_of(key: Key) -> list[str]:
    law, clusters, _, fit, _, _ = key
    return [
        f"{FAMILY}/{study.cell_name(fit, estimand, law, clusters, arm)}"
        for estimand in study.estimands_of(fit)
        for arm, _ in study.arms(fit, law, clusters, estimand)
    ]


def _cell_of(key: Key) -> str:
    law, clusters, _, fit, _, _ = key
    first = study.estimands_of(fit)[0]
    return f"{FAMILY}/{study.cell_name(fit, first, law, clusters, floor.REFERENCE_ARM)}"


def _replicate_of(key: Key) -> int:
    return key[2]


def _fit_key(key: Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    law, clusters, replicate, fit, budget, stream = key
    return study.fit_rows(fit, law, clusters, replicate, frame, budget=budget, stream=stream)


def keys(budget: int | None = None, stream: str = "property") -> list[Key]:
    """Every key of the grid: each fit's draws of each ``(law, J)``."""
    return [
        (law, clusters, replicate, fit, budget, stream)
        for fit, law, clusters in study.grid()
        for replicate in range(study.requested(fit, budget))
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


def _rows_by_cell(summary: pd.DataFrame) -> dict[str, pd.Series]:
    table = summary.loc[summary["property"] == FAMILY]
    return {str(row["cell"]): row for _, row in table.iterrows()}


def _pairs(
    summary: pd.DataFrame, family: str, estimands: Any
) -> dict[tuple[str, str, str, int], tuple[pd.Series, pd.Series]]:
    rows = _rows_by_cell(summary)
    out = {}
    for fit, law, clusters in study.grid():
        if study.FITS[fit] != family:
            continue
        for estimand in estimands(fit, law):
            positive = rows.get(study.cell_name(fit, estimand, law, clusters, floor.REFERENCE_ARM))
            control = rows.get(study.cell_name(fit, estimand, law, clusters, "iid_t_control"))
            if positive is not None and control is not None:
                out[(fit, law, estimand, clusters)] = (positive, control)
    return out


def floor_cells(
    summary: pd.DataFrame, family: str
) -> dict[tuple[str, str, str, int], tuple[pd.Series, pd.Series]]:
    """``(configuration, law, estimand, J)`` of each gated arm mean of ``family`` the summary
    holds, to its positive and control summary rows: the input of
    :func:`~tests.studies.few_cluster_floor.arm_mean_floor`."""
    return _pairs(summary, family, lambda fit, law: study.arm_means_of(fit))


def transfer_cells(
    summary: pd.DataFrame, family: str
) -> dict[tuple[str, str, str, int], tuple[pd.Series, pd.Series]]:
    """``(configuration, law, estimand, J)`` of each transfer-check contrast of ``family``."""

    def contrasts(fit: str, law: str) -> list[str]:
        return [
            name
            for name in study.estimands_of(fit)
            if name in study.CONTRASTS and study.transfer_check(fit, law, name)
        ]

    return _pairs(summary, family, contrasts)


def shipped_floor(family: str) -> int:
    """The family's shipped contrast floor (``CLUSTER_INTERVAL_FLOORS``)."""
    return int(CLUSTER_INTERVAL_FLOORS[floor.PACKAGE_FAMILIES[family]])


def source_hard_cell() -> floor.Counterpart:
    """S-a's committed primary row at the hard cell: its hits over its draws."""
    table = pd.read_csv(SOURCE_PERFORMANCE)
    _, estimand, _, _ = study.HARD_CELL
    row = table.loc[
        (table["implementation"] == "cleverly")
        & (table["scenario"] == SOURCE_PRIMARY)
        & (table["estimand"] == estimand)
    ].iloc[0]
    draws = int(row["replicates"])
    return floor.Counterpart(
        "few-cluster-floor-tmle", hits=round(float(row["coverage"]) * draws), draws=draws
    )


def hard_cell_name() -> str:
    """The positive cell of the known hard cell (:data:`~tests.studies.few_cluster_arm_means.HARD_CELL`)."""
    family, estimand, law, clusters = study.HARD_CELL
    fit = next(name for name, owner in study.FITS.items() if owner == family)
    return study.cell_name(fit, estimand, law, clusters, floor.REFERENCE_ARM)


def hard_cell_holds(summary: pd.DataFrame) -> bool:
    """Whether S-k's in-sample ``LTMLE`` ``ey_regimen[never]`` positive at 10 clusters on
    ``unequal_informative`` passes and agrees with S-a's primary there (plan 5.5): the two-sample
    99% interval of the coverage difference holds 0. A cell the summary lacks does not hold."""
    row = _rows_by_cell(summary).get(hard_cell_name())
    if row is None or not bool(row["passed"]):
        return False
    return floor.agrees_with_committed(row, source_hard_cell(), control=False)


def contrast_floor(family: str, summary: pd.DataFrame | None = None) -> int:
    """The lowest count ``arm_mean_floor`` may return for ``family``: its shipped contrast floor
    (the stacked one is S-j's, :math:`\\max(F, 4)` with :math:`F \\ge 4`), and for in-sample
    ``LTMLE`` :data:`HARD_CELL_CONTRAST_FLOOR` unless :func:`hard_cell_holds` (plan 5.5)."""
    base = shipped_floor(family)
    if family == study.HARD_CELL[0] and (summary is None or not hard_cell_holds(summary)):
        return max(base, HARD_CELL_CONTRAST_FLOOR)
    return base


def route(summary: pd.DataFrame, family: str) -> floor.Route:
    """The arm-mean floor :math:`F_m` of ``family``, its route and its red owners."""
    return floor.arm_mean_floor(
        floor_cells(summary, family),
        family=family,
        anchor=study.ANCHORS[family],
        contrast_floor=contrast_floor(family, summary),
        ladder=study.LADDERS[family],
        owner=study.RED_OWNER,
        prefix=FAMILY,
        clause_one_laws=study.ARM_MEAN_LAWS,
    )


def routes(summary: pd.DataFrame) -> dict[str, floor.Route]:
    """:func:`route` of every family."""
    return {family: route(summary, family) for family in study.LADDERS}


def transfer_owners(summary: pd.DataFrame) -> dict[str, tuple[str, str] | None]:
    """Each red transfer-check row and its owner (plan 5.4, the S-k transfer row).

    The contrast reads the registered rule on every law, clause 1 included. A red below the
    family's shipped contrast floor is held under :data:`~tests.studies.few_cluster_arm_means.RED_OWNER`
    with :func:`~tests.studies.few_cluster_floor.red_reading` (positive) or the control reading;
    a red at or above it is a defect suspect of that shipped floor (``None``).
    """
    out: dict[str, tuple[str, str] | None] = {}
    for family in study.LADDERS:
        threshold = shipped_floor(family)
        for key, (positive, control) in sorted(transfer_cells(summary, family).items()):
            for row, is_control in ((positive, False), (control, True)):
                if bool(row["passed"]):
                    continue
                name = f"{FAMILY}/{row['cell']}"
                if key[3] >= threshold:
                    out[name] = None
                elif is_control:
                    out[name] = (study.RED_OWNER, floor.CONTROL_READING)
                else:
                    out[name] = (study.RED_OWNER, floor.red_reading(positive))
    return out


# ------------------------------------------------------------------------------------- pilot


def gated_inputs(positive: pd.Series, control: pd.Series, law: str, estimand: str) -> dict:
    """The M5 sizing inputs of one gated cell: clause 1 counts on the arm-mean laws only (an
    arm mean on an ATE law has its ratio set to 0 with no spread, so clause 1 passes), and on
    every law for a transfer-check contrast."""
    inputs = floor.sizing_inputs(positive, control)
    if estimand not in study.CONTRASTS and law not in study.ARM_MEAN_LAWS:
        inputs.update(ratio=0.0, ratio_sd=0.0)
    return inputs


def positive_pass(inputs: Mapping[str, float], replicates: int) -> float:
    """G1 and G2's plug-in pass probability of a positive at ``replicates``."""
    return floor.g1_pass_probability(inputs["coverage"], replicates) * floor.bias_pass_probability(
        inputs["standardized_bias"], replicates
    )


def expected_floor(
    coverage: Mapping[int, Iterable[float]], *, ladder: Sequence[int], contrast: int
) -> int:
    """:math:`\\hat F_m`: the smallest ladder count, at least ``contrast``, from which every
    count's lowest pilot coverage is at least
    :data:`~tests.studies.few_cluster_floor.EXPECTED_FLOOR_COVERAGE`; unlike
    :func:`~tests.studies.few_cluster_floor.expected_floor` it is not capped at the anchor, and
    it is :data:`~cleverly._inference_status.FEW_CLUSTER_THRESHOLD` when the top count fails."""
    out = FEW_CLUSTER_THRESHOLD
    for count in sorted(ladder, reverse=True):
        if min(coverage[count]) < floor.EXPECTED_FLOOR_COVERAGE:
            break
        if count >= contrast:
            out = count
    return out


def predicted_floor(red: Iterable[int], *, ladder: Sequence[int], contrast: int) -> int:
    """The floor ``arm_mean_floor`` gives when exactly the counts ``red`` fail."""
    top = max(red, default=0)
    above = [count for count in sorted(ladder) if count >= contrast and count > top]
    return above[0] if above else FEW_CLUSTER_THRESHOLD


def family_sizing(summary: pd.DataFrame, family: str) -> dict[str, Any]:
    """One family's pilot sizing (M5 over its gated cells at or above :math:`\\hat F_m`, or at the top count),
    predicted floor and predicted reds."""
    contrast = contrast_floor(family, summary)
    ladder = study.LADDERS[family]
    arms = floor_cells(summary, family)
    transfer = transfer_cells(summary, family)
    inputs = {
        key: gated_inputs(positive, control, key[1], key[2])
        for key, (positive, control) in {**arms, **transfer}.items()
    }
    coverage: dict[int, list[float]] = {}
    for key in arms:
        coverage.setdefault(key[3], []).append(inputs[key]["coverage"])
    start = expected_floor(coverage, ladder=ladder, contrast=contrast)
    shipped = shipped_floor(family)
    # With no count expected to pass, the top count still decides between it and 40.
    sized = min(start, max(ladder))
    above = [
        values
        for key, values in inputs.items()
        if key[3] >= (shipped if key in transfer else sized)
    ]
    replicates = floor.required_replicates(above)
    plugin = floor.family_pass_probability(above, replicates)
    predictive = 1.0
    for values in above:
        predictive *= floor.predictive_g1(
            round(values["hits"]), int(values["pilot_draws"]), replicates
        )
    red_counts: set[int] = set()
    reds: dict[str, str] = {}
    for key, (positive, control) in sorted(arms.items()):
        values = inputs[key]
        if positive_pass(values, replicates) < pilot.RED_PROBABILITY:
            red_counts.add(key[3])
            reds[f"{FAMILY}/{positive['cell']}"] = floor.red_reading(positive)
        clause_two = pilot.control_power({**values, "ratio": 0.0, "ratio_sd": 0.0}, replicates)
        if clause_two < pilot.CONTROL_POWER:
            red_counts.add(key[3])
            reds[f"{FAMILY}/{control['cell']}"] = floor.CONTROL_READING
        elif key[1] not in study.ARM_MEAN_LAWS:
            full = pilot.control_power(floor.sizing_inputs(positive, control), replicates)
            if full < pilot.CONTROL_POWER:
                reds[f"{FAMILY}/{control['cell']}"] = floor.UNPOWERED_READING
        elif pilot.control_power(values, replicates) < pilot.CONTROL_POWER:
            red_counts.add(key[3])
            reds[f"{FAMILY}/{control['cell']}"] = floor.CONTROL_READING
    predicted = predicted_floor(red_counts, ladder=ladder, contrast=contrast)
    for key, (positive, control) in sorted(transfer.items()):
        values = inputs[key]
        below = key[3] < shipped
        if positive_pass(values, replicates) < pilot.RED_PROBABILITY:
            reading = floor.red_reading(positive) if below else DEFECT_SUSPECT
            reds[f"{FAMILY}/{positive['cell']}"] = reading
        if pilot.control_power(values, replicates) < pilot.CONTROL_POWER:
            reds[f"{FAMILY}/{control['cell']}"] = floor.CONTROL_READING if below else DEFECT_SUSPECT
    return {
        "expected_floor": start,
        "predicted_floor": predicted,
        "replicates": replicates,
        "plugin": plugin,
        "predictive": predictive,
        "target_met": plugin >= floor.FAMILY_PASS_TARGET,
        "reds": reds,
    }


def primary_coverage(summary: pd.DataFrame) -> dict[tuple[str, int], float]:
    """The pilot's all-draw coverage of each primary candidate."""
    rows = _rows_by_cell(summary)
    return {
        (estimand, clusters): float(
            rows[
                study.cell_name(
                    study.PRIMARY_FIT, estimand, study.PRIMARY_LAW, clusters, floor.REFERENCE_ARM
                )
            ]["all_draw_coverage"]
        )
        for estimand, clusters in study.PRIMARY_CANDIDATES
    }


def primary_pick(summary: pd.DataFrame) -> tuple[str, int]:
    """The declared primary rule applied to the pilot (:func:`~tests.studies.few_cluster_arm_means.primary_choice`)."""
    return study.primary_choice(primary_coverage(summary))


def hard_cell_prediction(summary: pd.DataFrame, replicates: int) -> tuple[float, float]:
    """The known hard cell's G1 and G2 plug-in pass probabilities at ``replicates`` (review F3)."""
    row = _rows_by_cell(summary)[hard_cell_name()]
    return (
        floor.g1_pass_probability(float(row["all_draw_coverage"]), replicates),
        floor.bias_pass_probability(float(row["standardized_bias"]), replicates),
    )


def pilot_sizing_table(summary: pd.DataFrame) -> pd.DataFrame:
    """``pilot-sizing.csv``: per family the sizing and predicted floor, then each predicted red;
    then the primary rule's pick and the hard cell's pass probabilities at its family's R."""
    out: list[dict[str, Any]] = []
    picked: dict[str, int] = {}
    for family in study.LADDERS:
        result = family_sizing(summary, family)
        picked[family] = int(result["replicates"])
        out.append(
            {
                "family": family,
                "kind": "family",
                "cell": "",
                "reading": "",
                **{key: value for key, value in result.items() if key != "reds"},
            }
        )
        out += [
            {"family": family, "kind": "predicted red", "cell": cell, "reading": reading}
            for cell, reading in result["reds"].items()
        ]
    estimand, clusters = primary_pick(summary)
    out.append(
        {
            "family": study.FITS[study.PRIMARY_FIT],
            "kind": "primary",
            "cell": study.cell_name(
                study.PRIMARY_FIT, estimand, study.PRIMARY_LAW, clusters, floor.REFERENCE_ARM
            ),
            "reading": "",
            "replicates": study.PRIMARY_REPLICATES,
        }
    )
    family = study.HARD_CELL[0]
    g1, g2 = hard_cell_prediction(summary, picked[family])
    out.append(
        {
            "family": family,
            "kind": "hard cell",
            "cell": hard_cell_name(),
            "reading": "",
            "replicates": picked[family],
            "g1_pass": g1,
            "g2_pass": g2,
        }
    )
    return pd.DataFrame(out)


def family_of(fit: str) -> str:
    """The floor family of a fit or of a cell's fit-and-estimand token."""
    return study.family_of(fit)


def timed_cell(cell: str) -> bool:
    """Whether a row carries its key's timing once: the first estimand's ``t_reference``."""
    fit, estimand, _, _, arm = study.parse(cell)
    return arm == floor.REFERENCE_ARM and estimand == study.estimands_of(fit)[0]


def synthetic_columns(cell: Any, rng: Any) -> dict[str, Any]:
    """The extra columns of a synthetic row: a shown interval at ``J - 2`` degrees of freedom."""
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


#: The property differential's declared size: each cell capped at this many replicates, on
#: :data:`DIFFERENTIAL_GROUPS` only, every estimand of each: an in-sample arm mean on an
#: arm-mean law, a d5 transfer check on an ATE law, DR-TMLE on an ATE law (clause 1 ungated)
#: and in-sample ``LTMLE`` at the hard cell.
DIFFERENTIAL_BUDGET: Final[int] = 200
DIFFERENTIAL_GROUPS: Final[tuple[tuple[str, str, int], ...]] = (
    ("tmle_in_sample_known", "equal10_arms", 6),
    ("tmle_stacked_d5_estimated", "unequal_informative", 10),
    ("drtmle_known", "equal10", 8),
    ("ltmle_in_sample", "unequal_informative", 10),
)


def differential_cells() -> frozenset[str]:
    """The ``family/cell`` names of :data:`DIFFERENTIAL_GROUPS`."""
    return frozenset(
        f"{FAMILY}/{study.cell_name(fit, estimand, law, clusters, arm)}"
        for fit, law, clusters in DIFFERENTIAL_GROUPS
        for estimand in study.estimands_of(fit)
        for arm, _ in study.arms(fit, law, clusters, estimand)
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
    "routes",
    "summarize_properties",
    "transfer_owners",
)
