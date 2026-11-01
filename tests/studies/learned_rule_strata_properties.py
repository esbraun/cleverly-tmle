"""Repeated-sampling properties of the learned-rule value inside baseline strata (X11 (m)).

Two families. Every row reads the error ``estimate - truth``, where the truth is the row's own
data-adaptive target, because the record declares ``truth_varies_by_replicate``.

``interval_calibration``
    Each draw of a setting fits the declared estimator at n = 1,000, runs the truth harness,
    and writes one positive row for each of the four parameters,
    ``<setting>_<stratum>__declared``, in four settings: the bounded (beta) outcome rows only
    and weighted, whose rule boundary lies in stratum 1, and the binary outcome rows only and
    weighted, whose boundary lies in stratum 2. The SE-ratio interval must lie inside 0.93 to
    1.07 and the coverage interval inside 0.92 to 0.98. Each positive carries the two derived
    controls of :func:`~tests.studies.evidence.property_verdicts.calibration_controls`, the SE
    shrunken by 0.70 and noise of one efficiency-bound unit, which check the instrument. Two
    construction controls, each on the same fit and each a standard-error control that must
    fall below the band, are reported apart:

    ``<stratum label>__unnormalised_curve_control``
        The plan's control: the stratum curve without its ``1 / P_{n,v}(S = s)``
        normalisation. Its SE is about the stratum share (0.2 to 0.45) times the reported one,
        so it fails by construction. It shows that the instrument resolves a gross
        normalisation error, not that the fold-local share is the right one, which the unit
        witnesses of E1 and of ``test_learned_rule_strata`` carry.
    ``<bounded setting>_all__share_dropped_control``
        The marginal curve without its share term ``(w_i / w_bar)(psi_S_i - psi)``. Its
        asymptotic SE ratio at the limit rule is 0.778 rows only and 0.773 weighted, so it
        tests the marginal derivation for the learned rule. Under the binary outcome the ratio
        is 0.98, so no binary label carries one.

``targeting_necessity``
    Each draw fits the bounded rows-only setting with E1's ``drop_w1_learner()``, which keeps
    ``A`` and ``W2`` of the design ``(A, W1, W2, S)``. That learner changes the rule as well as
    the regression: its plug-in rule is a function of ``W2`` alone. So each row's truth is the
    harness refit with the same learner, the value of the rules that fit learned. About 16% of
    the pilot's replicates learn a rule that leaves one ``W2`` level untreated in some fold, so
    the necessity target varies by replication too (truth SD 0.021 in stratum 0). For each
    stratum parameter the positive arm is the fit, and the control is the same fit without its
    fluctuation: the fold average of the initial stratum plug-ins at the fit's rule. The
    control learner is stratum-blind, so its plug-in is nearly the same in every stratum. The
    cell shows that targeting repairs a stratum-blind initial fit at a learned rule; it does not
    show repair of a subtle misspecification within a stratum.

``weak_margin_diagnostic``
    The weak-margin regime the declared binary law leaves (design record, D1): the binary
    rows-only law at E1's blip ``0.1 - 0.3 W1`` (``binary_weak``, E1's own draw) and at
    ``-0.6 + 1.5 W1`` (``binary_moderate``), :data:`WEAK_REPLICATES` draws each at n = 1,000.
    Each draw fits the declared estimator and the same estimator without ``strata=`` on the same
    sample and fold seed, so the same folds and fold rules, and runs the harness on both. One
    cell per parameter, ``<law>_<stratum>__diagnostic``, carries the role ``diagnostic``: no
    margin is read and no verdict forms. Each row carries the unstratified fit's marginal
    estimate, SE and truth (``unstratified_*``) and the margin measures. The summary publishes
    the coverage and the SE ratio of each cell with their intervals, the same for the
    unstratified marginal fit, and the mean boundary-stratum disagreement and treated share. It
    exists so that the page can state the weak-margin limit from an artifact, and that the
    unstratified learned-rule fit shares it.

A red ``interval_calibration`` cell is diagnosed before it is routed. The tightest positive is
``binary_steep_weighted_s2`` (pilot SE ratio 0.973, the cell with the most fold-rule
disagreement): a red there is read against this family and the oracle SE first, and the law is
not moved again after a verdict.

A property draw's sample seed is ``stream_seed(record, "property_sample", family, setting,
replicate, *suffix)`` and its fold seed the same tuple with ``"fold_partition"`` first; rule L3
of the RM30 studies (:func:`~tests.studies._learned_rule_law.seed_labels`) sets each suffix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _fold_evaluated_strata_law as strata
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_strata_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    cell_groups,
    control_row,
    ratio_intervals,
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
from tests.studies.learned_rule_strata import (
    CALIBRATION_LABELS,
    EFFICIENCY_BOUNDS,
    NECESSITY_LABELS,
    POSITIVE,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    SETTING_LABELS,
    SHARE_DROPPED,
    SHARE_DROPPED_LABELS,
    STRATUM_LABELS,
    STUDY,
    UNNORMALISED,
    WEAK_FAMILY,
    WEAK_KIND,
)

CALIBRATION_N = PRIMARY_N
#: The per-arm rule of :func:`~tests.studies.mtp_study_tools.per_arm_replicates` over the 16
#: positives, on the pilot read at its size, 4,000 draws per setting
#: (``test_learned_rule_strata_design``).
CALIBRATION_REPLICATES = 14_800
NECESSITY_REPLICATES = 3_600
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.5
WEAK_REPLICATES = 4_000

#: The two construction controls, each a below-band SE control of ``SE_CONTROL_SIDES``.
OWN_CONTROLS = (UNNORMALISED, SHARE_DROPPED)

#: The sample law of each setting: a weighted setting draws the same sample as its rows-only twin
#: (the weight is a column the fit reads), so rule L3 judges its seeds against that twin's.
SAMPLE_LAW = {
    "unweighted": "unweighted",
    "weighted": "unweighted",
    "binary_steep": "binary_steep",
    "binary_steep_weighted": "binary_steep",
}

#: One draw label per setting and family; each draw writes the rows of all four parameters.
PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    *(
        base.PropertyDraw(
            "interval_calibration", setting, CALIBRATION_REPLICATES, SAMPLE_LAW[setting]
        )
        for setting in law.SETTINGS
    ),
    base.PropertyDraw("targeting_necessity", "unweighted", NECESSITY_REPLICATES, "unweighted"),
    *(base.PropertyDraw(WEAK_FAMILY, name, WEAK_REPLICATES, name) for name in law.DIAGNOSTIC_LAWS),
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The columns a ``weak_margin_diagnostic`` row carries beside the harness columns: the paired
#: unstratified fit's marginal estimate, SE and truth. Every other row leaves them empty.
UNSTRATIFIED_COLUMNS = ("unstratified_estimate", "unstratified_std_error", "unstratified_truth")

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = (*law.HARNESS_COLUMNS, *UNSTRATIFIED_COLUMNS)

#: The descriptive summary columns of a ``weak_margin_diagnostic`` cell. No verdict reads them.
WEAK_SUMMARY = (
    "unstratified_coverage",
    "unstratified_coverage_ci_lower",
    "unstratified_coverage_ci_upper",
    "unstratified_se_ratio",
    "unstratified_se_ratio_ci_lower",
    "unstratified_se_ratio_ci_upper",
    "boundary_disagreement_mean",
    "boundary_treated_share_mean",
)

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, setting = key[0]
    return law.draw(setting, n, sample_seed)


def _cell_of(key: _Key) -> str:
    """The cell whose leading replicates the systematic prefix reads: the marginal value."""
    family, setting = key[0][:2]
    if family == "targeting_necessity":
        return f"{family}/{setting}_s0__targeted"
    if family == WEAK_FAMILY:
        return f"{family}/{setting}_all__{WEAK_KIND}"
    return f"{family}/{setting}_all__{POSITIVE}"


def _replicate_of(key: _Key) -> int:
    return int(key[0][2])


def _cells_of(key: _Key) -> list[str]:
    family, setting = key[0][:2]
    if family == "targeting_necessity":
        return [
            f"{family}/{label}__{arm}"
            for label in NECESSITY_LABELS
            for arm in ("targeted", "untargeted")
        ]
    if family == WEAK_FAMILY:
        return [f"{family}/{setting}_{label}__{WEAK_KIND}" for label in law.PARAMETER_LABELS]
    cells: list[str] = []
    for label in SETTING_LABELS[setting]:
        name = f"{family}/{label}__{POSITIVE}"
        cells.append(name)
        if label in STRATUM_LABELS:
            cells.append(f"{family}/{label}__{UNNORMALISED}")
        if label in SHARE_DROPPED_LABELS:
            cells.append(f"{family}/{label}__{SHARE_DROPPED}")
        cells.extend(DERIVED_CELLS[name])
    return cells


def _harness_columns(fitted: law.Measured, name: str, setting: str) -> dict[str, Any]:
    return {
        "oracle_se": fitted.truth.oracle_ses[name],
        "rule_rows_checked": fitted.truth.rows_checked,
        "solver_warnings": fitted.warnings,
        **law.boundary_measures(fitted.result, setting),
    }


def _positive(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    fitted: law.Measured,
    name: str,
    setting: str,
) -> dict[str, Any]:
    replicate, n, requested = where
    row = replicate_row(
        property_name=family,
        cell=cell,
        role="positive",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=fitted.truth.truths[name],
        estimate=fitted.result.estimates[name],
        alpha=STUDY.margins.alpha,
    )
    row.update(_harness_columns(fitted, name, setting))
    return row


def _control(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    fitted: law.Measured,
    name: str,
    estimate: float,
    standard_error: float,
    setting: str,
) -> dict[str, Any]:
    replicate, n, requested = where
    row = control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=fitted.truth.truths[name],
        estimate=estimate,
        standard_error=standard_error,
        critical=law.CRITICAL,
    )
    row.update(_harness_columns(fitted, name, setting))
    return row


def calibration_rows(
    fitted: law.Measured, setting: str, where: tuple[int, int, int]
) -> list[dict[str, Any]]:
    """The positive rows and the construction controls of one calibration fit."""
    family = "interval_calibration"
    rows: list[dict[str, Any]] = []
    for label in SETTING_LABELS[setting]:
        parameter_label = label.removeprefix(f"{setting}_")
        name = law.parameter(parameter_label)
        estimate = fitted.result.estimates[name]
        rows.append(_positive(family, f"{label}__{POSITIVE}", where, fitted, name, setting))
        if label in STRATUM_LABELS:
            rows.append(
                _control(
                    family,
                    f"{label}__{UNNORMALISED}",
                    where,
                    fitted,
                    name,
                    float(estimate.psi),
                    law.unnormalised_se(fitted.result, parameter_label),
                    setting,
                )
            )
        if label in SHARE_DROPPED_LABELS:
            rows.append(
                _control(
                    family,
                    f"{label}__{SHARE_DROPPED}",
                    where,
                    fitted,
                    name,
                    float(estimate.psi),
                    law.share_dropped_se(fitted.result),
                    setting,
                )
            )
    return rows


def necessity_rows(fitted: law.Measured, where: tuple[int, int, int]) -> list[dict[str, Any]]:
    """The targeted and untargeted rows of one targeting-necessity fit."""
    family = "targeting_necessity"
    rows: list[dict[str, Any]] = []
    for label in NECESSITY_LABELS:
        parameter_label = label.removeprefix("unweighted_")
        name = law.parameter(parameter_label)
        estimate = fitted.result.estimates[name]
        rows.append(_positive(family, f"{label}__targeted", where, fitted, name, "unweighted"))
        rows.append(
            _control(
                family,
                f"{label}__untargeted",
                where,
                fitted,
                name,
                law.untargeted(fitted.result, parameter_label),
                float(estimate.std_error),
                "unweighted",
            )
        )
    return rows


def paired(frame: pd.DataFrame, setting: str, fold_seed: int) -> tuple[law.Measured, law.Measured]:
    """The declared fit and the unstratified fit of one sample, on the same folds.

    The fold seed draws the same outer folds for both fits, which the fold index of each
    checks; a mismatch stops the run.
    """
    declared = law.measured(frame, setting, fold_seed)
    unstratified = law.measured(frame, setting, fold_seed, stratified=False)
    if not np.array_equal(
        strata.fold_index(declared.result), strata.fold_index(unstratified.result)
    ):
        raise RuntimeError("the unstratified fit drew other folds than the declared one")
    return declared, unstratified


def weak_margin_rows(
    fitted: tuple[law.Measured, law.Measured], setting: str, where: tuple[int, int, int]
) -> list[dict[str, Any]]:
    """The diagnostic rows of one weak-margin draw: the declared fit's four parameters, each
    with the paired unstratified fit's marginal estimate, SE and truth."""
    declared, unstratified = fitted
    replicate, n, requested = where
    other = unstratified.result.estimates[law.ESTIMAND]
    paired_columns = {
        "unstratified_estimate": float(other.psi),
        "unstratified_std_error": float(other.std_error),
        "unstratified_truth": unstratified.truth.truths[law.ESTIMAND],
    }
    rows: list[dict[str, Any]] = []
    for label in law.PARAMETER_LABELS:
        name = law.parameter(label)
        row = replicate_row(
            property_name=WEAK_FAMILY,
            cell=f"{setting}_{label}__{WEAK_KIND}",
            role=DIAGNOSTIC_ROLE,
            replicate=replicate,
            n=n,
            requested=requested,
            truth=declared.truth.truths[name],
            estimate=declared.result.estimates[name],
            alpha=STUDY.margins.alpha,
        )
        row.update(_harness_columns(declared, name, setting))
        row["solver_warnings"] = declared.warnings + unstratified.warnings
        row.update(paired_columns)
        rows.append(row)
    return rows


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, setting, replicate, n, requested, _, fold_seed, _ = key[0]
    where = (replicate, n, requested)
    if family == "interval_calibration":
        return calibration_rows(law.measured(frame, setting, fold_seed), setting, where)
    if family == "targeting_necessity":
        fitted = law.measured(frame, setting, fold_seed, learner=strata.drop_w1_learner())
        return necessity_rows(fitted, where)
    if family == WEAK_FAMILY:
        return weak_margin_rows(paired(frame, setting, fold_seed), setting, where)
    raise ValueError(f"unknown property family {family!r}")


def _payloads(record: StudyRecord, cap: int | None) -> list[_Key]:
    resolved = suffixes(record)
    out = []
    for draw in PROPERTY_DRAWS:
        suffix = resolved[draw.family, draw.label]
        replicates = draw.replicates if cap is None else min(cap, draw.replicates)
        for replicate in range(replicates):
            sample_seed = stream_seed(
                record, "property_sample", draw.family, draw.label, replicate, *suffix
            )
            fold_seed = stream_seed(
                record, "fold_partition", draw.family, draw.label, replicate, *suffix
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
    """Fit every property replication and return one row per cell and replication."""
    outcomes = fit_properties(
        _payloads(record, cap),
        _replication,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
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
        critical=law.CRITICAL,
        positive_suffix=POSITIVE,
    )
    for column in ("rule_rows_checked", "solver_warnings"):
        controls[column] = controls[column].astype(int)
    for column in UNSTRATIFIED_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    rows = pd.concat([rows, controls], ignore_index=True)
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


TARGETING_NECESSITY = property_verdicts.Necessity(
    family="targeting_necessity",
    labels=NECESSITY_LABELS,
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
    truth_varies=STUDY.truth_varies_by_replicate,
)


def report_weak_margin(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Report each ``weak_margin_diagnostic`` cell and the paired unstratified fit; no verdict.

    The summary already holds the declared fit's coverage with its exact interval. This adds
    the bootstrap interval of its SE ratio and, on every cell of a law, the unstratified fit's
    marginal coverage and SE ratio with the same intervals on the same draws, and the mean
    boundary-stratum disagreement and treated share. It writes ``passed`` and
    ``property_passed`` ``True`` on every diagnostic row, meaning the numbers were computed and
    are published; the published role says that no margin was read.
    """
    family = summary["property"] == WEAK_FAMILY
    roles = set(summary.loc[family, "role"].astype(str))
    if family.any() and roles != {DIAGNOSTIC_ROLE}:
        raise ValueError(f"{WEAK_FAMILY} publishes roles {sorted(roles)}, not diagnostic")
    margins = STUDY.margins
    groups = cell_groups(rows, WEAK_FAMILY)
    for cell in sorted(summary.loc[family, "cell"]):
        group = groups[cell]
        unstratified = group.assign(
            estimate=group["unstratified_estimate"],
            std_error=group["unstratified_std_error"],
            truth=group["unstratified_truth"],
        )
        mask = family & (summary["cell"] == cell)
        for prefix, frame, labels in (
            ("", group, (cell,)),
            ("unstratified_", unstratified, (cell, "unstratified")),
        ):
            ratio = ratio_intervals(
                frame,
                replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, WEAK_FAMILY, *labels),
                truth_varies=STUDY.truth_varies_by_replicate,
            )["se_ratio"]
            summary.loc[mask, f"{prefix}se_ratio_ci_lower"] = ratio.low
            summary.loc[mask, f"{prefix}se_ratio_ci_upper"] = ratio.high
        error = (unstratified["estimate"] - unstratified["truth"]).to_numpy(float)
        std_error = unstratified["std_error"].to_numpy(float)
        covered = np.abs(error) <= law.CRITICAL * std_error
        coverage = clopper_pearson(
            int(covered.sum()), int(covered.size), confidence_level=margins.confidence_level
        )
        summary.loc[mask, "unstratified_coverage"] = float(np.mean(covered))
        summary.loc[mask, "unstratified_coverage_ci_lower"] = coverage.low
        summary.loc[mask, "unstratified_coverage_ci_upper"] = coverage.high
        summary.loc[mask, "unstratified_se_ratio"] = float(
            np.mean(std_error) / np.std(error, ddof=1)
        )
        for column in law.MARGIN_COLUMNS:
            summary.loc[mask, f"{column}_mean"] = float(group[column].mean())
        summary.loc[mask, "passed"] = True
        summary.loc[mask, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, and the report of each diagnostic cell.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", *WEAK_SUMMARY),
        rate_labels=(),
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    report_weak_margin(summary, rows)
    return finish(summary, rates)


#: The declared rules of the failure accounting: the shared calibration rule, named here
#: because the module declares its own rules, and the targeting-necessity rules. The
#: weak-margin diagnostic writes its verdict columns without reading a margin, so it has none.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
