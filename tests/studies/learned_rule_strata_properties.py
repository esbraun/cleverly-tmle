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
    harness refit with the same learner, the value of the rules that fit learned. For each
    stratum parameter the positive arm is the fit, and the control is the same fit without its
    fluctuation: the fold average of the initial stratum plug-ins at the fit's rule. The
    control learner is stratum-blind, so its plug-in is nearly the same in every stratum. The
    cell shows that targeting repairs a stratum-blind initial fit at a learned rule; it does not
    show repair of a subtle misspecification within a stratum.

A property draw's sample seed is ``stream_seed(record, "property_sample", family, setting,
replicate, *suffix)`` and its fold seed the same tuple with ``"fold_partition"`` first; rule L3
of the RM30 studies (:func:`~tests.studies._learned_rule_law.seed_labels`) sets each suffix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _fold_evaluated_strata_law as strata
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_strata_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
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
)

CALIBRATION_N = PRIMARY_N
#: The per-arm rule of :func:`~tests.studies.mtp_study_tools.per_arm_replicates` over the 16
#: positives, on the pilot read at its size, 4,000 draws per setting
#: (``test_learned_rule_strata_design``).
CALIBRATION_REPLICATES = 14_800
NECESSITY_REPLICATES = 8_000
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.5

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
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The columns every property row carries beside the shared schema.
ROW_COLUMNS = law.HARNESS_COLUMNS

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, setting = key[0]
    return law.draw(setting, n, sample_seed)


def _cell_of(key: _Key) -> str:
    """The cell whose leading replicates the systematic prefix reads: the marginal value."""
    family, setting = key[0][:2]
    if family == "targeting_necessity":
        return f"{family}/{setting}_s0__targeted"
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


def _harness_columns(fitted: law.Measured, name: str) -> dict[str, Any]:
    return {
        "oracle_se": fitted.truth.oracle_ses[name],
        "rule_rows_checked": fitted.truth.rows_checked,
        "solver_warnings": fitted.warnings,
    }


def _positive(
    family: str, cell: str, where: tuple[int, int, int], fitted: law.Measured, name: str
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
    row.update(_harness_columns(fitted, name))
    return row


def _control(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    fitted: law.Measured,
    name: str,
    estimate: float,
    standard_error: float,
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
    row.update(_harness_columns(fitted, name))
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
        rows.append(_positive(family, f"{label}__{POSITIVE}", where, fitted, name))
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
        rows.append(_positive(family, f"{label}__targeted", where, fitted, name))
        rows.append(
            _control(
                family,
                f"{label}__untargeted",
                where,
                fitted,
                name,
                law.untargeted(fitted.result, parameter_label),
                float(estimate.std_error),
            )
        )
    return rows


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, setting, replicate, n, requested, _, fold_seed, _ = key[0]
    where = (replicate, n, requested)
    if family == "interval_calibration":
        return calibration_rows(law.measured(frame, setting, fold_seed), setting, where)
    if family == "targeting_necessity":
        fitted = law.measured(frame, setting, fold_seed, learner=strata.drop_w1_learner())
        return necessity_rows(fitted, where)
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


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, every statistic on the error of each row.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("targeting_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    return finish(summary, rates)


#: The declared rules of the failure accounting: the shared calibration rule, named here
#: because the module declares its own rules, and the targeting-necessity rules.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
