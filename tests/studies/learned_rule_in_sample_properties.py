"""Repeated-sampling properties of the in-sample learned-rule value.

Four families, in the order the declaration reads them. Every fit is in sample
(``cross_fit=False``) at n = 2,000.

``interval_calibration``
    The declared fit of each law (``<law>__declared``), with the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    :data:`SHRUNKEN_SE_FACTOR`, and noise of one efficiency-bound unit added to each estimate.
    Each ``non_exceptional`` draw also writes ``non_exceptional__interpolating_control``: the
    clipped interpolating tree fitted in sample on the same draw, read against the value of its
    own rule. It is adaptive, so condition C5a fails, and its SE-ratio interval must fall below
    the band.
``targeting_necessity``
    An outcome learner without the ``W1`` main effect (``WithoutW1MainLogistic``, marked
    parametric), with the correct treatment learner. Its limit rule is far from exceptional.
    The positive arm is the fit. The control is the same fit without its fluctuation: the mean
    of the initial plug-in at the fit's own rule. Bias only.
``weak_limit_necessity``
    The same pair with the plan's ``drop_w1_learner`` (marked parametric), whose limit blip is
    too weak at n = 2,000 for its one rule to settle (role ``diagnostic``, no verdict): the
    weak-margin diagnostic of the ``targeting_necessity`` learner choice (the design record,
    D2). It is the study's evidence of what condition C3a buys. The summary publishes the
    paired displacement as ``weak_limit_displacement``, the share of draws whose rule differs
    from the limit rule as ``weak_limit_flip_share``, and the targeted arm's mean error on
    settled and on flipped draws, in its error SD over every draw, as
    ``weak_limit_settled_bias`` and ``weak_limit_flipped_bias``.
``target_distinction``
    The declared estimate and interval of every calibration draw of each law
    (``<law>__fold_average``), read against the fold-average target of the same draw's 10-fold
    rules (role ``diagnostic``, no margin and no verdict). Each row carries the full-sample
    truth and the grid disagreement of the fold rules with the full-sample rule; the summary
    publishes the mean truth gap, its SD over the declared cell's error SD, and the mean
    disagreement.

Every statistic reads the error ``estimate - truth`` of each row, because the record declares
``truth_varies_by_replicate``. A property draw's sample seed is
``stream_seed(record, "property_sample", family, label, replicate, *suffix)``, its fit seed the
same tuple with ``"fit_seed"`` first, and the diagnostic's fold seed the same tuple with
``"fold_partition"`` first; rule L3 of the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_in_sample_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    cell_groups,
    control_row,
    paired_displacement,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
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
from tests.studies.learned_rule_in_sample import (
    EFFICIENCY_BOUNDS,
    INTERPOLATING,
    POSITIVE,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    REPORTING_CELLS,
    REPORTING_FAMILY,
    STUDY,
    WEAK_LIMIT_FAMILY,
    WEAK_LIMIT_LABEL,
    WEAK_LIMIT_RULE_VALUE,
)

base = law.base

CALIBRATION_LABELS = law.SCENARIOS
CALIBRATION_N = PRIMARY_N
#: The per-arm rule of :func:`~tests.studies.mtp_study_tools.per_arm_replicates` over the two
#: positives, on the 4,000-draw pilot read at the declared size: family-wise 0.956,
#: ``misspecified_limit`` sets it (``test_learned_rule_in_sample_design``).
CALIBRATION_REPLICATES = 7_200
#: The ``targeting_necessity`` draws: the smallest multiple of 400 whose predictive pass
#: probability of the positive arm's bias leg is at least 0.95 on the 4,000-draw pilot of its
#: learner (0.959). The control's bias and the displacement pass with probability 1 there
#: (``test_learned_rule_in_sample_design``). The weak-limit diagnostic draws as many.
NECESSITY_REPLICATES = 1_200
#: The plan's shrunken-SE control: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80
#: The ``targeting_necessity`` displacement floor, in positive-arm SDs (the RM30 value).
TARGETING_DISPLACEMENT = 0.5
LAW = "non_exceptional"

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    *(
        base.PropertyDraw(
            "interval_calibration", f"{label}__{POSITIVE}", CALIBRATION_REPLICATES, label
        )
        for label in CALIBRATION_LABELS
    ),
    base.PropertyDraw("targeting_necessity", LAW, NECESSITY_REPLICATES, LAW),
    base.PropertyDraw(WEAK_LIMIT_FAMILY, WEAK_LIMIT_LABEL, NECESSITY_REPLICATES, LAW),
)

#: The learner, the arm roles and the cell label of each necessity-shaped family. Both draw
#: the ``non_exceptional`` law.
NECESSITY_KINDS = {
    "targeting_necessity": ("without_w1_main", ("positive", "control"), LAW),
    WEAK_LIMIT_FAMILY: ("drop_w1", (DIAGNOSTIC_ROLE, DIAGNOSTIC_ROLE), WEAK_LIMIT_LABEL),
}


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The cells a calibration draw writes beside its positive arm: the reporting cell of its law,
#: and on ``non_exceptional`` the interpolating control.
INTERPOLATING_CELL = f"interval_calibration/{LAW}__{INTERPOLATING}"
REPORTING = {
    label: f"{REPORTING_FAMILY}/{cell}"
    for label, cell in zip(CALIBRATION_LABELS, REPORTING_CELLS, strict=True)
}

#: The diagnostic columns of a reporting row. Every other row leaves them empty.
REPORTING_COLUMNS = ("full_sample_truth", "grid_disagreement")

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*law.HARNESS_COLUMNS, *REPORTING_COLUMNS)

#: The descriptive summary columns of the reporting cells. No verdict reads them.
REPORTING_SUMMARY = ("mean_truth_gap", "truth_gap_sd_ratio", "mean_grid_disagreement")

#: The descriptive summary columns of the weak-limit pair. No verdict reads them.
WEAK_LIMIT_SUMMARY = (
    "weak_limit_displacement",
    "weak_limit_flip_share",
    "weak_limit_settled_bias",
    "weak_limit_flipped_bias",
)

#: A ``drop_w1`` draw's rule flipped when its truth differs from the limit rule's value by more
#: than this. The harness integrates the limit rule exactly as :data:`WEAK_LIMIT_RULE_VALUE`
#: does, so a settled draw's truth equals it to rounding, and the other three rules a
#: ``drop_w1`` fit can learn sit at least 7e-3 away (``test_learned_rule_in_sample_design``).
FLIP_TOLERANCE = 1e-6

#: The integer harness columns, which a derived control must keep integer.
INTEGER_COLUMNS = ("rule_rows_checked", "solver_warnings", "in_sample_warnings")

_Key = tuple[tuple[str, str, int, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, _, scenario = key[0]
    return base.draw(scenario, n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    family, label = key[0][:2]
    if family in NECESSITY_KINDS:
        stem = NECESSITY_KINDS[family][2]
        return [f"{family}/{stem}__targeted", f"{family}/{stem}__untargeted"]
    name = f"{family}/{label}"
    scenario = label.removesuffix(f"__{POSITIVE}")
    written = [name, REPORTING[scenario]]
    if scenario == LAW:
        written.insert(1, INTERPOLATING_CELL)
    return [*written, *DERIVED_CELLS.get(name, ())]


def _row(
    family: str,
    cell: str,
    role: str,
    where: tuple[int, int, int],
    truth: float,
    estimate: Any,
    columns: dict[str, Any],
) -> dict[str, Any]:
    replicate, n, requested = where
    row = replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    row.update(columns)
    return row


def reporting_row(
    declared: tuple[Any, base.Truth, law.Counts],
    scenario: str,
    fold_seed: int,
    where: tuple[int, int, int],
) -> dict[str, Any]:
    """The declared fit read against the fold-average target of the same draw (no verdict)."""
    result, truth, counts = declared
    average = law.fold_average(result, scenario, fold_seed)
    columns = law.harness_columns(truth, counts)
    columns.update(full_sample_truth=truth.truth, grid_disagreement=average.grid_disagreement)
    return _row(
        REPORTING_FAMILY,
        REPORTING[scenario].split("/", 1)[1],
        DIAGNOSTIC_ROLE,
        where,
        average.truth,
        result.estimates[law.ESTIMAND],
        columns,
    )


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, n, requested, _, fit_seed, fold_seed, scenario = key[0]
    where = (replicate, n, requested)
    if family in NECESSITY_KINDS:
        kind, (positive, control), stem = NECESSITY_KINDS[family]
        result, truth, counts = law.measured(frame, scenario, kind, fit_seed)
        estimate = result.estimates[law.ESTIMAND]
        columns = law.harness_columns(truth, counts)
        targeted = _row(
            family, f"{stem}__targeted", positive, where, truth.truth, estimate, columns
        )
        untargeted = control_row(
            property_name=family,
            cell=f"{stem}__untargeted",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth.truth,
            estimate=law.untargeted_estimate(result),
            standard_error=float(estimate.std_error),
            critical=base.CRITICAL,
            role=control,
        )
        untargeted.update(columns)
        return [targeted, untargeted]
    if family != "interval_calibration":
        raise ValueError(f"unknown property family {family!r}")
    fitted = law.measured(frame, scenario, "declared", fit_seed)
    result, truth, counts = fitted
    columns = law.harness_columns(truth, counts)
    rows = [
        _row(family, label, "positive", where, truth.truth, result.estimates[law.ESTIMAND], columns)
    ]
    if scenario == LAW:
        tree, tree_truth, tree_counts = law.measured(frame, scenario, "interpolating", fit_seed)
        rows.append(
            _row(
                family,
                f"{LAW}__{INTERPOLATING}",
                "control",
                where,
                tree_truth.truth,
                tree.estimates[law.ESTIMAND],
                law.harness_columns(tree_truth, tree_counts),
            )
        )
    rows.append(reporting_row(fitted, scenario, fold_seed, where))
    return rows


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            seeds = [
                stream_seed(record, stream, draw.family, draw.label, replicate, *suffix)
                for stream in ("property_sample", "fit_seed", "fold_partition")
            ]
            out.append(
                (
                    (
                        draw.family,
                        draw.label,
                        replicate,
                        CALIBRATION_N,
                        replicates,
                        *seeds,
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
    for column in REPORTING_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    controls = calibration_controls(
        rows,
        record,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_BOUNDS,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=base.CRITICAL,
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
    family="targeting_necessity",
    labels=(LAW,),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
    truth_varies=STUDY.truth_varies_by_replicate,
)


def report_target_distinction(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Report the declared estimate against the fold-average target of the same draws.

    The summary already holds the bias, coverage and SE ratio against that target. This adds
    the mean of the truth gap ``Psi_{d_n} - (1/V) sum_v Psi_{d_v}``, its SD over the SD of the
    declared cell's error on the same draws, and the mean grid disagreement. It writes
    ``passed`` and ``property_passed`` ``True``, meaning the numbers were computed and are
    published; the published role says that no margin was read.
    """
    family = summary["property"] == REPORTING_FAMILY
    roles = set(summary.loc[family, "role"].astype(str))
    if family.any() and roles != {DIAGNOSTIC_ROLE}:
        raise ValueError(f"{REPORTING_FAMILY} publishes roles {sorted(roles)}, not diagnostic")
    groups = cell_groups(rows, REPORTING_FAMILY)
    for cell in sorted(summary.loc[family, "cell"]):
        group = groups[cell]
        gap = (group["full_sample_truth"] - group["truth"]).to_numpy(float)
        error = (group["estimate"] - group["full_sample_truth"]).to_numpy(float)
        mask = family & (summary["cell"] == cell)
        summary.loc[mask, "mean_truth_gap"] = float(np.mean(gap))
        summary.loc[mask, "truth_gap_sd_ratio"] = float(np.std(gap, ddof=1) / np.std(error, ddof=1))
        summary.loc[mask, "mean_grid_disagreement"] = float(group["grid_disagreement"].mean())
        summary.loc[mask, "passed"] = True
        summary.loc[mask, "property_passed"] = True


def flipped_draws(truth: pd.Series) -> pd.Series:
    """Whether each ``drop_w1`` draw's rule differs from its limit rule, read from its truth."""
    return (truth.astype(float) - WEAK_LIMIT_RULE_VALUE).abs() > FLIP_TOLERANCE


def report_weak_limit(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Report the weak-limit pair: its displacement and flips, and ``passed`` with no margin read.

    The summary already holds each arm's bias, standardized bias, coverage and SE ratio. This
    adds, on both rows of the pair:

    - ``weak_limit_displacement``: the paired displacement of the untargeted arm from the
      targeted one, in the targeted arm's error SD;
    - ``weak_limit_flip_share``: the share of draws whose rule differs from the learner's limit
      rule, which treats every row (truth not :data:`WEAK_LIMIT_RULE_VALUE`);
    - ``weak_limit_settled_bias`` and ``weak_limit_flipped_bias``: the targeted arm's mean
      error on the settled and on the flipped draws, over its error SD on every draw, so the
      two average to the arm's standardized bias with the flip share as weight. A subset with
      no draw publishes NaN.

    It writes ``passed`` and ``property_passed`` ``True``: the numbers were computed and are
    published.
    """
    family = summary["property"] == WEAK_LIMIT_FAMILY
    if not family.any():
        return
    roles = set(summary.loc[family, "role"].astype(str))
    if roles != {DIAGNOSTIC_ROLE}:
        raise ValueError(f"{WEAK_LIMIT_FAMILY} publishes roles {sorted(roles)}, not diagnostic")
    summary.loc[family, "weak_limit_displacement"] = paired_displacement(
        rows,
        WEAK_LIMIT_FAMILY,
        f"{WEAK_LIMIT_LABEL}__targeted",
        f"{WEAK_LIMIT_LABEL}__untargeted",
        truth_varies=STUDY.truth_varies_by_replicate,
    )
    targeted = cell_groups(rows, WEAK_LIMIT_FAMILY)[f"{WEAK_LIMIT_LABEL}__targeted"]
    error = (targeted["estimate"] - targeted["truth"]).to_numpy(float)
    flipped = flipped_draws(targeted["truth"]).to_numpy()
    spread = float(np.std(error, ddof=1))

    def subset_bias(mask: np.ndarray) -> float:
        return float(np.mean(error[mask]) / spread) if mask.any() else float("nan")

    summary.loc[family, "weak_limit_flip_share"] = float(np.mean(flipped))
    summary.loc[family, "weak_limit_settled_bias"] = subset_bias(~flipped)
    summary.loc[family, "weak_limit_flipped_bias"] = subset_bias(flipped)
    summary.loc[family, "passed"] = True
    summary.loc[family, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, and the report of the two diagnostic families.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", *WEAK_LIMIT_SUMMARY, *REPORTING_SUMMARY),
        rate_labels=(),
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    report_target_distinction(summary, rows)
    report_weak_limit(summary, rows)
    return finish(summary, rates)


#: The declared rules of the failure accounting: the shared calibration rule, which reads the
#: interpolating control through ``SE_CONTROL_SIDES``, and the necessity rule. The two diagnostic
#: families write their verdict columns without reading a margin, so they have none.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
