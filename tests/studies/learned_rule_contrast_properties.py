"""Repeated-sampling properties of the learned rule beside known regimes, and their contrasts.

Five families, in the order the declaration reads them. Every fit is at n = 2,000 beside
``Static(1)`` and ``Static(0)``, and a law draw fits both reports on one fold seed.

``interval_calibration``
    ``non_exceptional``: each report's five parameters (``<law>_<report>_<tag>__declared``),
    each with the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the SE shrunken by
    :data:`SHRUNKEN_SE_FACTOR`, and noise of one efficiency-bound unit added to each estimate.
``conservative_coverage``
    ``misspecified_limit``: each report's five parameters, held to the bias leg and to a
    coverage interval whose lower bound is at or above the calibration floor, with no SE-ratio
    gate (Corollary 3's route). Each positive's control is the same estimate with its SE times
    :data:`~tests.studies.learned_rule_contrast.COVERAGE_SHRINK`, whose coverage interval must
    lie wholly below the floor.
``contrast_covariance``
    Each contrast of both laws and reports with the SE ``|se_L - se_K|`` it would have if the two
    estimates had correlation one. Its coverage interval must lie wholly below
    :data:`~tests.studies.learned_rule_contrast.CORRELATION_ONE_CEILING`.
``independent_se_diagnostic``
    The same contrasts with ``sqrt(se_L^2 + se_K^2)``, the SE if the two were independent
    (role ``diagnostic``, no verdict).
``targeting_necessity``
    RM30's ``drop_w1_learner`` on ``non_exceptional``, fold-evaluated, read on the contrast
    against ``Static(0)``: the targeted contrast against the untargeted difference of the
    initial plug-ins.

Every statistic reads the error ``estimate - truth`` of each row, because the record declares
``truth_varies_by_replicate``. A property draw's sample seed is
``stream_seed(record, "property_sample", family, label, replicate, *suffix)`` and its fold seed
the same tuple with ``"fold_partition"`` first; rule L3 of the RM30 studies sets each suffix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_contrast_law as law
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
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
from tests.studies.learned_rule_contrast import (
    CALIBRATION_LABELS,
    CORRELATION_ONE,
    CORRELATION_ONE_CEILING,
    COVARIANCE_FAMILY,
    COVERAGE_CONTROL,
    COVERAGE_FAMILY,
    COVERAGE_LAW,
    COVERAGE_SHRINK,
    EFFICIENCY_BOUNDS,
    INDEPENDENT,
    INDEPENDENT_FAMILY,
    NECESSITY_FAMILY,
    NECESSITY_LABEL,
    NECESSITY_TAG,
    POSITIVE,
    PRIMARY_LAW,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    REPORTS,
    STUDY,
    covariance_cells,
    harness_columns,
    independent_cells,
)

base = law.base

CALIBRATION_N = PRIMARY_N
#: The ``non_exceptional`` draws: the arm-pooled rule of
#: :func:`~tests.studies.mtp_study_tools.replicates_from_pilot` over the five fold-evaluated
#: positives, on the 4,000-draw pilot read at the declared size (family-wise 0.951; the per-arm
#: rule gives the same count). Each stacked positive is the same estimate as its fold-evaluated
#: one at ten equal folds, with an SE within 0.03% of it, so it passes or fails with it and is no
#: separate arm of the pooled model (the design record, D2; ``test_learned_rule_contrast_design``).
CALIBRATION_REPLICATES = 5_600
#: The ``misspecified_limit`` draws: the smallest multiple of 400 whose predictive probability
#: that every ``conservative_coverage`` cell passes (each positive's bias leg and coverage floor,
#: each control's coverage ceiling) is at least 0.95 on the 4,000-draw pilot, over the five
#: fold-evaluated arms for the reason of :data:`CALIBRATION_REPLICATES` (0.964; 2,800 gives 0.943;
#: ``test_learned_rule_contrast_design``).
COVERAGE_REPLICATES = 3_200
#: The ``targeting_necessity`` draws: the study's floor of 1,000 draws for a gated cell, as a
#: multiple of 400. The bias leg of the positive arm would pass with probability 0.977 at 400 on
#: the 4,000-draw pilot of its learner, and the control's bias and the displacement sit far past
#: their thresholds (``test_learned_rule_contrast_design``).
NECESSITY_REPLICATES = 1_200
#: The plan's shrunken-SE control of the calibration family: the declared SE times 0.8.
SHRUNKEN_SE_FACTOR = 0.80
#: The ``targeting_necessity`` displacement floor, in positive-arm SDs (the RM30 value).
TARGETING_DISPLACEMENT = 0.5

PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    base.PropertyDraw("interval_calibration", PRIMARY_LAW, CALIBRATION_REPLICATES, PRIMARY_LAW),
    base.PropertyDraw(COVERAGE_FAMILY, COVERAGE_LAW, COVERAGE_REPLICATES, COVERAGE_LAW),
    base.PropertyDraw(
        NECESSITY_FAMILY, f"{PRIMARY_LAW}_drop_w1", NECESSITY_REPLICATES, PRIMARY_LAW
    ),
)

#: The family each law's draws file their positives under.
LAW_FAMILY = {PRIMARY_LAW: "interval_calibration", COVERAGE_LAW: COVERAGE_FAMILY}


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = law.HARNESS_COLUMNS

#: The integer harness columns, which a derived control must keep integer.
INTEGER_COLUMNS = ("rule_rows_checked", "solver_warnings")

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, scenario = key[0]
    return base.draw(scenario, n, sample_seed)


def law_cells(scenario: str) -> list[str]:
    """Every ``property/cell`` one draw of ``scenario``'s law family writes, controls included."""
    family = LAW_FAMILY[scenario]
    kinds = (POSITIVE,) if family == "interval_calibration" else (POSITIVE, COVERAGE_CONTROL)
    written = [
        f"{family}/{scenario}_{report}_{tag}__{kind}"
        for report in REPORTS
        for tag in law.TAGS
        for kind in kinds
    ]
    written += [f"{COVARIANCE_FAMILY}/{cell}" for cell in covariance_cells(scenario)]
    written += [f"{INDEPENDENT_FAMILY}/{cell}" for cell in independent_cells(scenario)]
    for cell in list(written):
        written += DERIVED_CELLS.get(cell, ())
    return written


def _cells_of(key: _Key) -> list[str]:
    family = key[0][0]
    if family == NECESSITY_FAMILY:
        return [f"{NECESSITY_FAMILY}/{cell}" for cell in STUDY.property_cells[NECESSITY_FAMILY]]
    # A law draw's label is its law.
    return law_cells(key[0][1])


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


def _control(
    family: str,
    cell: str,
    role: str,
    where: tuple[int, int, int],
    truth: float,
    estimate: float,
    standard_error: float,
    columns: dict[str, Any],
) -> dict[str, Any]:
    replicate, n, requested = where
    row = control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        standard_error=standard_error,
        critical=base.CRITICAL,
        role=role,
    )
    row.update(columns)
    return row


def law_rows(fitted: law.Measured, scenario: str, where: tuple[int, int, int]) -> list[dict]:
    """The rows one law draw writes: each report's five parameters and their direct controls."""
    family = LAW_FAMILY[scenario]
    rows = []
    for report in REPORTS:
        result = fitted.results[report]
        truths = law.truths(result, fitted.refit, scenario)
        for tag, name in law.TAGS.items():
            estimate = result.estimates[name]
            truth, oracle = truths[tag]
            columns = harness_columns(fitted, oracle)
            stem = f"{scenario}_{report}_{tag}"
            rows.append(
                _row(family, f"{stem}__{POSITIVE}", "positive", where, truth, estimate, columns)
            )
            point = float(estimate.psi)
            if family == COVERAGE_FAMILY:
                rows.append(
                    _control(
                        family,
                        f"{stem}__{COVERAGE_CONTROL}",
                        "control",
                        where,
                        truth,
                        point,
                        COVERAGE_SHRINK * float(estimate.std_error),
                        columns,
                    )
                )
            if tag in law.CONTRAST_TAGS:
                rows.append(
                    _control(
                        COVARIANCE_FAMILY,
                        f"{stem}__{CORRELATION_ONE}",
                        "control",
                        where,
                        truth,
                        point,
                        law.correlation_one_se(result, tag),
                        columns,
                    )
                )
                rows.append(
                    _control(
                        INDEPENDENT_FAMILY,
                        f"{stem}__{INDEPENDENT}",
                        DIAGNOSTIC_ROLE,
                        where,
                        truth,
                        point,
                        law.independent_se(result, tag),
                        columns,
                    )
                )
    return rows


def necessity_rows(fitted: law.Measured, where: tuple[int, int, int]) -> list[dict]:
    """The targeted and untargeted contrast against ``Static(0)`` of one ``drop_w1`` fit."""
    result = fitted.results["fold_evaluated"]
    truth, oracle = law.truths(result, fitted.refit, PRIMARY_LAW)[NECESSITY_TAG]
    estimate = result.estimates[law.TAGS[NECESSITY_TAG]]
    columns = harness_columns(fitted, oracle)
    targeted, untargeted = STUDY.property_cells[NECESSITY_FAMILY]
    return [
        _row(NECESSITY_FAMILY, targeted, "positive", where, truth, estimate, columns),
        _control(
            NECESSITY_FAMILY,
            untargeted,
            "control",
            where,
            truth,
            law.untargeted(result, NECESSITY_TAG),
            float(estimate.std_error),
            columns,
        ),
    ]


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, _, replicate, n, requested, _, fold_seed, scenario = key[0]
    where = (replicate, n, requested)
    if family == NECESSITY_FAMILY:
        fitted = law.measured(
            frame, scenario, base.drop_w1_learner(), fold_seed, ("fold_evaluated",)
        )
        return necessity_rows(fitted, where)
    if family != LAW_FAMILY[scenario]:
        raise ValueError(f"unknown property family {family!r} for {scenario}")
    fitted = law.measured(frame, scenario, base.outcome_learner(), fold_seed)
    return law_rows(fitted, scenario, where)


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            sample_seed, fold_seed = (
                stream_seed(record, stream, draw.family, draw.label, replicate, *suffix)
                for stream in ("property_sample", "fold_partition")
            )
            out.append(
                (
                    (
                        draw.family,
                        draw.label,
                        replicate,
                        CALIBRATION_N,
                        replicates,
                        sample_seed,
                        fold_seed,
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
    family=NECESSITY_FAMILY,
    labels=(NECESSITY_LABEL,),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
    truth_varies=STUDY.truth_varies_by_replicate,
)


def coverage_verdicts(summary: pd.DataFrame, *, floor: float) -> None:
    """The ``conservative_coverage`` verdicts: a one-sided coverage claim, and its control.

    A positive passes when its bias is established inside the margin and the lower bound of its
    exact coverage interval is at or above ``floor``; its SE ratio is published and read by no
    verdict, because the claim admits a conservative interval. A control passes when its coverage
    interval lies wholly below ``floor``.
    """
    family = summary["property"] == COVERAGE_FAMILY
    positive = family & (summary["role"] == "positive")
    control = family & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"].astype(bool) & (
        summary.loc[positive, "coverage_ci_lower"] >= floor
    )
    summary.loc[control, "passed"] = summary.loc[control, "coverage_ci_upper"] < floor


def covariance_verdicts(summary: pd.DataFrame, *, ceiling: float) -> None:
    """The ``contrast_covariance`` verdicts: each correlation-one control must under-cover."""
    family = summary["property"] == COVARIANCE_FAMILY
    summary.loc[family, "passed"] = summary.loc[family, "coverage_ci_upper"] < ceiling


def report_independent(summary: pd.DataFrame) -> None:
    """Publish the independent-SE diagnostic: ``passed`` with no margin read.

    The summary already holds each cell's SE ratio and coverage. ``passed`` and
    ``property_passed`` read ``True``, meaning the numbers were computed and are published; the
    published role says that no margin was read.
    """
    family = summary["property"] == INDEPENDENT_FAMILY
    roles = set(summary.loc[family, "role"].astype(str))
    if family.any() and roles != {DIAGNOSTIC_ROLE}:
        raise ValueError(f"{INDEPENDENT_FAMILY} publishes roles {sorted(roles)}, not diagnostic")
    summary.loc[family, "passed"] = True
    summary.loc[family, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, and the report of the diagnostic family.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("targeting_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    coverage_verdicts(summary, floor=STUDY.margins.calibration_coverage[0])
    covariance_verdicts(summary, ceiling=CORRELATION_ONE_CEILING)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    report_independent(summary)
    return finish(summary, rates)


def _coverage_floor(ctx: pf.Context) -> float:
    return ctx.margins.calibration_coverage[0]


#: The ``conservative_coverage`` rule of :func:`coverage_verdicts`.
COVERAGE_RULE = pf.FamilyRule(
    families=frozenset({COVERAGE_FAMILY}),
    shared="coverage_verdicts",
    clauses=(
        pf.bias_clause("bias equivalent", kind="equivalent", applies=pf.roles("positive")),
        pf.proportion_clause(
            "coverage at or above the calibration floor",
            statistic="S2",
            judge=pf.floor_judge(_coverage_floor),
            applies=pf.roles("positive"),
        ),
        pf.proportion_clause(
            "shrunken coverage below the calibration floor",
            statistic="S2",
            judge=pf.below_judge(_coverage_floor),
            applies=pf.roles("control"),
        ),
    ),
)

#: The ``contrast_covariance`` rule of :func:`covariance_verdicts`.
COVARIANCE_RULE = pf.FamilyRule(
    families=frozenset({COVARIANCE_FAMILY}),
    shared="covariance_verdicts",
    clauses=(
        pf.proportion_clause(
            "correlation-one coverage below its ceiling",
            statistic="S2",
            judge=pf.below_judge(lambda ctx: CORRELATION_ONE_CEILING),
            applies=pf.roles("control"),
        ),
    ),
)

#: The declared rules of the failure accounting: the shared calibration rule, the two module
#: rules and the necessity rule. The diagnostic family writes its verdict columns without
#: reading a margin, so it has none.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    COVERAGE_RULE,
    COVARIANCE_RULE,
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
