r"""The ``few_cluster_reference`` family of ``few-cluster-floor-stacked`` (X54 part (j)).

Each key fits one configuration of :data:`tests.studies.few_cluster_floor_stacked.SOURCES` on
one draw of a ``(law, J)`` cell, ``stream_seed(STUDY, "property", f"{law}/J{J}", replicate)``
(the pilot reads ``"pilot"``), and writes one row per arm
(:func:`tests.studies.few_cluster_floor_stacked.fit_rows`). Every configuration of a
``(law, J, replicate)`` reads the same sample, so the fixed-bounds arm and the slack pair on the
draw. A fit that raises is a lost key, a miss under the failure accounting.

**Gated** (the registered two-clause rule, ``few_cluster_rule(shown_only=True)``): G1 and G2 on
every ``t_reference`` row, the fixed-bounds arm's included; both control clauses on the
``iid_t_control`` row of each default-bounds configuration. Part (c) is (C) alone: each
``few_cluster_candidate`` pair passes when the 99% lower end of its paired coverage gain over
(A) is at least -0.03 (S7), and each configuration's ``few_cluster_candidate_pooled`` row from
:math:`\max(F, 10) = 10` when its pooled gain lies above -0.01.

**Floor and outcome.** :func:`route` reads the family's floor over the default-bounds
configurations by plan A 5.8 with ``declared_route(redeclaration=False)``: a red at or above 10
is a defect suspect, read as chance only after a recorded no-defect diagnosis and agreement with
the committed ``clustered-few-cluster-tmle`` stacked cell, which only the two-covariate known
configuration has at 10, 20 and 30 clusters; every other red there reads ``support withdrawn``.
:func:`fixed_bounds_reading` reads the fixed-bounds arm against that floor (plan 1.2 item 4),
:func:`jackknife_status` the stacked jackknife (plan A 5.7 rules 1, 4 and 5), and
:func:`outcome` folds the three into plan 5.4's consequence.

**Tail readings** (:data:`TAIL_COLUMNS`, on each ``t_reference`` row, read by no verdict): the
median and the 10%-trimmed SE ratio with 99% bootstrap intervals beside the mean's
(``se_ratio_ci_*``); the largest SE over the empirical SD and its 99th and 99.9th percentiles;
the share of draws whose fit reached a propensity bound, with the shown-only coverage and SE
ratio split by it; the paired estimated-to-known mean-SE ratio; and on a fixed-bounds row the
paired ``"auto"``-minus-fixed-bounds estimate in SD units with its standard error.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_pilot as pilot
from tests.studies import few_cluster_floor_stacked as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import Interval, percentile_interval
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, bootstrap_draw_blocks
from tests.studies.evidence.property_failures import cell_frame, shown_mask
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
#: ``<fit>__jackknife``.
Key = tuple[str, int, int, str, "int | None", str]
JACKKNIFE_SUFFIX: Final[str] = "__jackknife"
#: The share each tail of the trimmed SE ratio drops.
TRIM: Final[float] = 0.10
#: The tail readings of each ``t_reference`` row.
TAIL_COLUMNS: Final[tuple[str, ...]] = (
    "se_ratio_median",
    "se_ratio_median_ci_lower",
    "se_ratio_median_ci_upper",
    "se_ratio_trimmed",
    "se_ratio_trimmed_ci_lower",
    "se_ratio_trimmed_ci_upper",
    "se_over_sd_max",
    "se_over_sd_q99",
    "se_over_sd_q999",
    "bound_reached_share",
    "shown_coverage_bound_reached",
    "shown_coverage_bound_clear",
    "shown_se_ratio_bound_reached",
    "shown_se_ratio_bound_clear",
    "paired_known_se_ratio",
    "fixed_bounds_shift_sd",
    "fixed_bounds_shift_se",
)
#: The reading of a predicted red at or above the anchor that no committed cell can compare.
DEFECT_SUSPECT: Final[str] = "defect suspect"


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


#: The arms of one fit, size law and count (a fixed-bounds group holds a positive and no gated
#: control); each (C) cell with (A)'s copy; and each pooled row alone.
PROPERTY_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    FAMILY: _groups(),
    floor.CANDIDATE_FAMILY: _candidate_groups(),
    floor.POOLED_FAMILY: tuple((name,) for name in study.POOLED_CELLS),
}
#: The module's rule for the part (c) families and the registered rule in its shown-only form.
PROPERTY_RULES = (floor.candidate_rule(), property_verdicts.few_cluster_rule(shown_only=True))


def _cell_truths(law: str) -> dict[str, float]:
    return {study.TARGET: study.source.law_truth(law)}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published replication cell, with its role, size and requested replicates."""
    out = []
    for fit, law, clusters in study.grid():
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
                    dgp=floor.CellLaw(f"{law} J{clusters}", functools.partial(_cell_truths, law)),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=10 * clusters,
                    replicates=study.requested(candidate=candidate),
                    seed=study.draw_seed(law, clusters, 0),
                    role=role,
                    estimand=study.TARGET,
                )
            )
    return tuple(out)


def _split(name: str) -> tuple[str, bool]:
    if name.endswith(JACKKNIFE_SUFFIX):
        return name.removesuffix(JACKKNIFE_SUFFIX), True
    return name, False


def _draw(key: Key) -> pd.DataFrame:
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
    arm = floor.JACKKNIFE_ARM if jackknife else floor.REFERENCE_ARM
    family = floor.CANDIDATE_FAMILY if jackknife else FAMILY
    return f"{family}/{study.cell_name(fit, law, clusters, arm)}"


def _replicate_of(key: Key) -> int:
    return key[2]


def _fit_key(key: Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    law, clusters, replicate, name, budget, stream = key
    fit, jackknife = _split(name)
    rows = study.jackknife_rows if jackknife else study.fit_rows
    return rows(fit, law, clusters, replicate, frame, budget=budget, stream=stream)


def keys(budget: int | None = None, stream: str = "property") -> list[Key]:
    """Every key of the grid: each fit's main keys, then (C)'s on the first 2,000 draws."""
    out: list[Key] = []
    for fit, law, clusters in study.grid():
        for replicate in range(study.requested(budget)):
            out.append((law, clusters, replicate, fit, budget, stream))
        if study.candidates(fit, clusters):
            for replicate in range(study.requested(budget, candidate=True)):
                out.append((law, clusters, replicate, fit + JACKKNIFE_SUFFIX, budget, stream))
    return out


#: The row columns beside the shared replicate columns.
ROW_COLUMNS: Final[tuple[str, ...]] = (*floor.EXTRA_COLUMNS, study.BOUND_COLUMN)


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
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *ROW_COLUMNS, *timing]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary, the registered few-cluster rule, the floor and tail readings, (C)."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            *FEW_CLUSTER_COLUMNS,
            *floor.READING_COLUMNS,
            *floor.CANDIDATE_COLUMNS,
            *TAIL_COLUMNS,
        ),
        rate_labels=(),
    )
    few_cluster_reference_verdicts(summary, rows, STUDY, PROPERTY_GROUPS[FAMILY], shown_only=True)
    floor.floor_readings(summary, rows, STUDY)
    tail_readings(summary, rows, STUDY)
    floor.candidate_verdicts(summary, rows, STUDY, PROPERTY_GROUPS[floor.CANDIDATE_FAMILY])
    rates.extend(floor.pooled_rows(summary, rows, STUDY, pooled_starts()))
    return finish(summary, rates)


# ------------------------------------------------------------------------------ tail readings


def _ratio_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    replicates: int,
    level: float,
    seed: int,
) -> Interval:
    """The percentile bootstrap of ``statistic(SE) / SD(estimate)`` over the shown draws."""
    rng = np.random.default_rng(seed)
    ratios = np.concatenate(
        [
            statistic(block[:, :, 0]) / block[:, :, 1].std(axis=1, ddof=1)
            for block in bootstrap_draw_blocks(values, replicates=replicates, rng=rng)
        ]
    )
    return percentile_interval(ratios[np.isfinite(ratios)], confidence_level=level)


def _median(se: np.ndarray) -> np.ndarray:
    return np.median(se, axis=-1)


def _trimmed(se: np.ndarray) -> np.ndarray:
    return np.asarray(stats.trim_mean(se, TRIM, axis=-1))


def _shown_ratio(shown: pd.DataFrame) -> float:
    if len(shown) < 2:
        return math.nan
    spread = float(np.std(shown["estimate"].to_numpy(dtype=float), ddof=1))
    return float(shown["std_error"].mean() / spread) if spread > 0 else math.nan


def _tail(group: pd.DataFrame, record: Any, cell: str) -> dict[str, float]:
    """The tail readings of one ``t_reference`` cell's rows."""
    shown = group.loc[shown_mask(group)]
    out: dict[str, float] = {}
    flags = pd.to_numeric(group[study.BOUND_COLUMN], errors="coerce")
    if flags.notna().any():
        out["bound_reached_share"] = float(flags.dropna().mean())
        shown_flags = flags.loc[shown.index]
        for label, mask in (("reached", shown_flags == 1), ("clear", shown_flags == 0)):
            part = shown.loc[mask]
            out[f"shown_coverage_bound_{label}"] = (
                float(part["covered"].mean()) if len(part) else math.nan
            )
            out[f"shown_se_ratio_bound_{label}"] = _shown_ratio(part)
    if len(shown) < 2:
        return out
    se = shown["std_error"].to_numpy(dtype=float)
    estimate = shown["estimate"].to_numpy(dtype=float)
    spread = float(np.std(estimate, ddof=1))
    if not spread > 0:
        return out
    scaled = se / spread
    out.update(
        se_over_sd_max=float(scaled.max()),
        se_over_sd_q99=float(np.quantile(scaled, 0.99)),
        se_over_sd_q999=float(np.quantile(scaled, 0.999)),
        se_ratio_median=float(np.median(se) / spread),
        se_ratio_trimmed=float(stats.trim_mean(se, TRIM) / spread),
    )
    values = np.column_stack([se, estimate])
    margins = record.margins
    for name, statistic in (("median", _median), ("trimmed", _trimmed)):
        interval = _ratio_interval(
            values,
            statistic,
            replicates=margins.bootstrap_replicates,
            level=margins.confidence_level,
            seed=stream_seed(record, "tail", name, cell),
        )
        out[f"se_ratio_{name}_ci_lower"] = interval.low
        out[f"se_ratio_{name}_ci_upper"] = interval.high
    return out


def _paired(left: pd.DataFrame, right: pd.DataFrame, column: str) -> pd.DataFrame:
    joined = pd.concat(
        [
            left.set_index("replicate")[column].rename("left"),
            right.set_index("replicate")[column].rename("right"),
        ],
        axis=1,
        join="inner",
    )
    return joined.astype(float)


def paired_known_se_ratio(estimated: pd.DataFrame, known: pd.DataFrame) -> float:
    """The mean SE of an estimated configuration over the known one's, on draws both show."""
    both = _paired(estimated.loc[shown_mask(estimated)], known.loc[shown_mask(known)], "std_error")
    return float(both["left"].mean() / both["right"].mean()) if len(both) else math.nan


def fixed_bounds_shift(default: pd.DataFrame, fixed: pd.DataFrame) -> tuple[float, float]:
    """The paired ``"auto"``-minus-fixed-bounds estimate over the default fit's empirical SD,
    and its standard error, on the draws both kept (the truncation part of the bias)."""
    both = _paired(default, fixed, "estimate").dropna()
    spread = float(np.std(default["estimate"].to_numpy(dtype=float), ddof=1))
    if len(both) < 2 or not spread > 0:
        return math.nan, math.nan
    difference = (both["left"] - both["right"]).to_numpy()
    return (
        float(difference.mean() / spread),
        float(difference.std(ddof=1) / math.sqrt(len(difference)) / spread),
    )


def tail_readings(summary: pd.DataFrame, rows: pd.DataFrame, record: Any) -> None:
    """Fill :data:`TAIL_COLUMNS` on every ``t_reference`` row of the family the summary holds."""
    family = summary["property"] == FAMILY
    for index in summary.index[family.to_numpy()]:
        name = str(summary.loc[index, "cell"])
        fit, law, clusters, arm = floor.parse(name)
        if arm != floor.REFERENCE_ARM:
            continue
        group = cell_frame(rows, FAMILY, name)
        if group is None:
            raise KeyError(name)
        values = _tail(group, record, name)
        known = cell_frame(
            rows, FAMILY, study.cell_name(study.KNOWN_OF.get(fit, fit), law, clusters, arm)
        )
        if fit in study.KNOWN_OF and known is not None:
            values["paired_known_se_ratio"] = paired_known_se_ratio(group, known)
        default = cell_frame(
            rows,
            FAMILY,
            study.cell_name(study.FIXED_BOUNDS_FITS.get(fit, fit), law, clusters, arm),
        )
        if fit in study.FIXED_BOUNDS_FITS and default is not None:
            shift, error = fixed_bounds_shift(default, group)
            values.update(fixed_bounds_shift_sd=shift, fixed_bounds_shift_se=error)
        for column, value in values.items():
            summary.loc[index, column] = value


# ---------------------------------------------------------------------------- floor, outcome


def _rows_by_cell(summary: pd.DataFrame, family: str = FAMILY) -> dict[str, pd.Series]:
    table = summary.loc[summary["property"] == family]
    return {str(row["cell"]): row for _, row in table.iterrows()}


def floor_cells(summary: pd.DataFrame) -> dict[tuple[str, str, int], tuple[pd.Series, pd.Series]]:
    """``(configuration, law, J)`` of each default-bounds cell the summary holds, to its positive
    and control summary rows."""
    rows = _rows_by_cell(summary)
    out = {}
    for fit in study.CONFIGURATIONS:
        for law in study.laws_of(fit):
            for clusters in study.LADDER:
                positive = rows.get(study.cell_name(fit, law, clusters, floor.REFERENCE_ARM))
                control = rows.get(study.cell_name(fit, law, clusters, "iid_t_control"))
                if positive is not None and control is not None:
                    out[(fit, law, clusters)] = (positive, control)
    return out


def fixed_bounds_cells(summary: pd.DataFrame) -> dict[tuple[str, str, int], pd.Series]:
    """``(fit, law, J)`` of each fixed-bounds cell the summary holds, to its positive row."""
    rows = _rows_by_cell(summary)
    out = {}
    for fit in study.FIXED_BOUNDS_FITS:
        for law in study.laws_of(fit):
            for clusters in study.LADDER:
                positive = rows.get(study.cell_name(fit, law, clusters, floor.REFERENCE_ARM))
                if positive is not None:
                    out[(fit, law, clusters)] = positive
    return out


def known_counterparts() -> dict[str, floor.Counterpart]:
    """The committed ``clustered-few-cluster-tmle`` stacked cells, keyed as the two-covariate
    known configuration's cells: the only configuration with a committed counterpart (5.1)."""
    out = {}
    for name, counterpart in floor.COMMITTED_CELLS.items():
        committed, rest = name.split("__", 1)
        if committed == floor.COUNTERPART_FITS["tmle_stacked_known"]:
            out[f"stacked_known__{rest}"] = counterpart
    return out


def route(summary: pd.DataFrame) -> floor.Route:
    """The family's floor, route and red owners over the default-bounds configurations."""
    return floor.declared_route(
        study.FLOOR_FAMILY,
        floor_cells(summary),
        anchor=study.ANCHOR,
        counterparts=known_counterparts(),
        redeclaration=False,
    )


def fixed_bounds_reading(
    summary: pd.DataFrame, family_floor: int
) -> tuple[tuple[int, ...], dict[str, tuple[str, str] | None]]:
    """The counts at or above ``family_floor`` where a fixed-bounds positive is red, and its owners.

    A red below the floor is held under :data:`~tests.studies.few_cluster_floor.RED_OWNER` by
    :func:`~tests.studies.few_cluster_floor.red_reading`. A red at or above it is no reading's:
    below 10 it scopes the lowered floor to a known mechanism or ``"auto"`` (plan 1.2 item 4,
    5.4), and from 10 it is a defect suspect, as any red there.
    """
    red: set[int] = set()
    owners: dict[str, tuple[str, str] | None] = {}
    for (_, _, clusters), positive in sorted(fixed_bounds_cells(summary).items()):
        if bool(positive["passed"]):
            continue
        name = f"{FAMILY}/{positive['cell']}"
        if clusters < family_floor:
            owners[name] = (floor.RED_OWNER, floor.red_reading(positive))
        else:
            owners[name] = None
            red.add(clusters)
    return tuple(sorted(red)), owners


def pooled_starts() -> dict[tuple[str, str], int]:
    """(C)'s admissibility count per configuration: :math:`\\max(F, 10)`, which is 10, because
    the declared route never sets the floor above the anchor."""
    return {(fit, floor.JACKKNIFE_ARM): study.ANCHOR for fit in study.CONFIGURATIONS}


def pooled_verdicts(summary: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """Each pooled row's verdict, by ``(configuration, candidate)``."""
    table = summary.loc[summary["property"] == floor.POOLED_FAMILY]
    return {
        tuple(str(row["cell"]).split("__")): bool(row["property_passed"])  # type: ignore[misc]
        for _, row in table.iterrows()
    }


def candidate_cells(summary: pd.DataFrame, rows: pd.DataFrame) -> list[floor.CandidateCell]:
    """(C)'s part (c) reading in each cell: paired counts, calibration, length and clause (i)."""
    pairs = rows.loc[rows["property"] == floor.CANDIDATE_FAMILY]
    frames = dict(tuple(pairs.groupby("cell", sort=False)))
    verdicts = _rows_by_cell(summary, floor.CANDIDATE_FAMILY)
    draws = floor.CANDIDATE_DRAWS
    out = []
    for fit, law, clusters in study.grid():
        for candidate in study.candidates(fit, clusters):
            name = study.cell_name(fit, law, clusters, candidate)
            empty = pairs.iloc[0:0]
            frame = frames.get(name, empty)
            control = frames.get(
                study.cell_name(fit, law, clusters, floor.reference_for(candidate)), empty
            )
            shown = frame.loc[(frame["replicate"] < draws) & (frame["withheld"] == 0)]
            length, high = math.nan, math.nan
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
                    else math.nan,
                )
            )
    return out


def jackknife_status(summary: pd.DataFrame, rows: pd.DataFrame) -> str:
    """The stacked jackknife: ``"option"`` when (C) is admissible from 10 clusters (plan A 5.7
    rule 1, clause (i) per cell and clause (ii) pooled per configuration, d5 included), else
    ``"refused: {reason}"``. Rule 2 is not re-run: S-a's (A) stands (P3), so rule 4 reads
    ``"option"`` for an admissible (C)."""
    admissible, reason = floor.admissibility(
        candidate_cells(summary, rows),
        floor.JACKKNIFE_ARM,
        start=study.ANCHOR,
        pooled=pooled_verdicts(summary),
    )
    return "option" if admissible else f"refused: {reason}"


def outcome(summary: pd.DataFrame, rows: pd.DataFrame) -> dict[str, Any]:
    """Plan 5.4's S-j consequence: the shipped floor, the bounds it holds for, the jackknife.

    ``floor`` is the family floor :math:`F` (10 unless every count from a lower one passes).
    Below 10, ``bounds`` is ``"auto or known"`` when a fixed-bounds positive is red at a count at
    or above :math:`F` (plan 1.2 item 4), else ``"at least (0.01, 0.99)"``; at 10 it is
    ``"any"``, the default floor every fit reads.
    """
    found = route(summary)
    red, _ = fixed_bounds_reading(summary, found.floor)
    if found.floor >= study.ANCHOR:
        scope = "any"
    else:
        scope = "auto or known" if red else "at least (0.01, 0.99)"
    return {
        "floor": found.floor,
        "route": found.route,
        "bounds": scope,
        "fixed_bounds_red": red,
        "jackknife": jackknife_status(summary, rows),
    }


# ------------------------------------------------------------------------------------- pilot


def positive_inputs(positive: pd.Series) -> dict[str, float]:
    """The sizing inputs of a positive with no gated control (the fixed-bounds arm): its control
    figures pass with certainty (:func:`~tests.studies.few_cluster_floor.control_pass_probability`
    at a zero SD reads 1 for a ratio below 0.80 and an SE ratio at most 1.20)."""
    requested = float(positive["requested_draws"])
    coverage = float(positive["all_draw_coverage"])
    return {
        "coverage": coverage,
        "hits": coverage * requested,
        "standardized_bias": float(positive["standardized_bias"]),
        "ratio": 0.0,
        "ratio_sd": 0.0,
        "se_ratio": 0.0,
        "se_ratio_sd": 0.0,
        "pilot_draws": requested,
    }


def sizing(summary: pd.DataFrame) -> dict[str, floor.FamilySizing]:
    """The family's M5 sizing off a pilot summary, the fixed-bounds positives included."""
    cells: dict[tuple[str, str, int], dict[str, float]] = {
        key: floor.sizing_inputs(positive, control)
        for key, (positive, control) in floor_cells(summary).items()
    }
    cells |= {key: positive_inputs(row) for key, row in fixed_bounds_cells(summary).items()}
    return {study.FLOOR_FAMILY: floor.family_sizing(cells, anchor=study.ANCHOR)}


def positive_pass(inputs: dict[str, float], replicates: int) -> float:
    """G1 and G2's plug-in pass probability of a positive at ``replicates``."""
    return floor.g1_pass_probability(inputs["coverage"], replicates) * floor.bias_pass_probability(
        inputs["standardized_bias"], replicates
    )


def _above_reading(cell: str) -> str:
    return floor.CHANCE_READING if cell in known_counterparts() else floor.SUPPORT_WITHDRAWN


def predicted_reds(summary: pd.DataFrame, replicates: int) -> dict[str, str]:
    """Each predicted red ``family/cell`` at ``replicates`` and the reading its position gives.

    A positive whose G1 and G2 figure is below 0.90, a control whose power under both clauses
    is below 0.90. Below the anchor: :func:`~tests.studies.few_cluster_floor.red_reading` or
    the control reading. At or above it: chance for a cell with a committed counterpart (held
    only once diagnosed), ``support withdrawn`` for any other.
    """
    out = {}
    cells: list[tuple[int, pd.Series, pd.Series | None]] = [
        (key[2], positive, control) for key, (positive, control) in floor_cells(summary).items()
    ]
    cells += [(key[2], positive, None) for key, positive in fixed_bounds_cells(summary).items()]
    for clusters, positive, control in sorted(cells, key=lambda item: str(item[1]["cell"])):
        inputs = (
            positive_inputs(positive) if control is None else floor.sizing_inputs(positive, control)
        )
        if positive_pass(inputs, replicates) < pilot.RED_PROBABILITY:
            reading = (
                floor.red_reading(positive)
                if clusters < study.ANCHOR
                else _above_reading(str(positive["cell"]))
            )
            out[f"{FAMILY}/{positive['cell']}"] = reading
        if control is not None and pilot.control_power(inputs, replicates) < pilot.CONTROL_POWER:
            reading = (
                floor.CONTROL_READING
                if clusters < study.ANCHOR
                else _above_reading(str(control["cell"]))
            )
            out[f"{FAMILY}/{control['cell']}"] = reading
    return out


def pilot_sizing_table(summary: pd.DataFrame) -> pd.DataFrame:
    """``pilot-sizing.csv``: the family's M5 sizing, then each predicted red and its reading."""
    result = sizing(summary)[study.FLOOR_FAMILY]
    out: list[dict[str, Any]] = [
        {
            "family": study.FLOOR_FAMILY,
            "kind": "family",
            "cell": "",
            "reading": "",
            "expected_floor": result.expected_floor,
            "replicates": result.replicates,
            "plugin": result.plugin,
            "predictive": result.predictive,
            "target_met": result.plugin >= floor.FAMILY_PASS_TARGET,
        }
    ]
    for cell, reading in predicted_reds(summary, result.replicates).items():
        out.append(
            {
                "family": study.FLOOR_FAMILY,
                "kind": "predicted red",
                "cell": cell,
                "reading": reading,
            }
        )
    return pd.DataFrame(out)


def family_of(fit: str) -> str:
    """The floor family of a fit, for the pilot's timing and candidate tables."""
    del fit
    return study.FLOOR_FAMILY


def synthetic_columns(cell: Any, rng: Any) -> dict[str, Any]:
    """The extra columns of a synthetic row: a shown interval, and a bound flag on a positive."""
    clusters = cell.n // 10
    return {
        "withheld": 0,
        "trigger": "",
        "reference_df": float(clusters - 2),
        "clusters": clusters,
        "half_width": float("nan"),
        "failed_refits": 0,
        "jackknife_error": "",
        study.BOUND_COLUMN: float(rng.uniform() < 0.1) if cell.role == "positive" else math.nan,
    }


#: The property differential's declared size: each cell capped at this many replicates, on
#: :data:`DIFFERENTIAL_GROUPS` only, one group per construction and every arm kind: a
#: below-anchor known pair, the d5 estimated pair at the anchor with its (C) pair and pooled
#: row, a fixed-bounds positive, and the estimated pair at the top count with its (C) pair.
DIFFERENTIAL_BUDGET: Final[int] = 200
DIFFERENTIAL_GROUPS: Final[tuple[tuple[str, str, int], ...]] = (
    ("stacked_known", "equal10", 4),
    ("stacked_d5_estimated", "unequal_informative", 10),
    ("stacked_d5_estimated_fixed_bounds", "unequal_informative", 10),
    ("stacked_estimated", "equal10", 39),
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
#: The property failure accounting's bound (:data:`tests.studies.few_cluster_floor.FAILURE_BOUND`).
FAILURE_BOUND = floor.FAILURE_BOUND


__all__: Sequence[str] = (
    "PROPERTY_GROUPS",
    "PROPERTY_RULES",
    "declared_cells",
    "fixed_bounds_reading",
    "generate_property_rows",
    "jackknife_status",
    "outcome",
    "route",
    "sizing",
    "summarize_properties",
)
