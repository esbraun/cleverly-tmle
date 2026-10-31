"""Repeated-sampling properties of fold-evaluated CV-TMLE with baseline strata (X28).

Two families, in the order the declaration reads them. Every cell's truth is the fixed
quadrature value of its parameter (:func:`~tests.studies._fold_evaluated_strata_law.truths`).

``interval_calibration``
    Each draw of a setting fits the declared estimator at n = 1,000 and writes one positive row
    for each of the 12 parameters, ``<setting>_<target>_<stratum>__declared``: the SE-ratio
    interval must lie inside 0.93 to 1.07 and the coverage interval inside 0.92 to 0.98. Each
    positive carries the two derived controls of
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls`, the SE shrunken by
    0.70 and noise of one efficiency-bound unit, which check the instrument. Two controls check
    the construction, each on the same fit and each read as a standard-error control that must
    fall below the band:

    ``<stratum label>__unnormalised_curve_control``
        The plan's control: the stratum curve without its ``1 / P_{n,v}(S = s)``
        normalisation. Its SE is about the stratum share (0.2 to 0.45) times the reported one,
        so it fails by a wide margin and discriminates weakly.
    ``<setting>_{ate,ey1}_all__share_dropped_control``
        The marginal curve without its share term ``(w_i / w_bar)(psi_S_i - psi)``. Its
        asymptotic SE ratio is 0.854 for ``ate`` and 0.656 to 0.660 for ``ey[1]``, so it tests
        the marginal derivation. The ``ey[0]`` share term moves the ratio to about 0.95 only,
        which the declared count cannot resolve, so that label carries no such control.

``targeting_necessity``
    Each draw fits the rows-only setting with RM30's ``drop_w1_learner()`` on the quasibinomial
    solver, which keeps ``A`` and ``W2`` of the design ``(A, W1, W2, S)`` and so drops ``S``
    too. For each stratum
    parameter the positive arm is the fit, and the control is the same fit without its
    fluctuation: the fold average of the initial stratum plug-ins. The control must be biased
    outside the margin, the positive inside, and the least displaced pair must move by at least
    :data:`TARGETING_DISPLACEMENT` positive-arm SDs.

A property draw's sample seed is ``stream_seed(record, "property_sample", family, setting,
replicate, *suffix)`` and its fold seed the same tuple with ``"fold_partition"`` first; rule L3
of the RM30 studies (:func:`~tests.studies._learned_rule_law.seed_labels`) sets each suffix.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _fold_evaluated_strata_law as law
from tests.studies import _learned_rule_law as base
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
from tests.studies.fold_evaluated_strata import (
    CALIBRATION_LABELS,
    EFFICIENCY_BOUNDS,
    NECESSITY_LABELS,
    POSITIVE,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    PROPERTY_CELLS,
    SHARE_DROPPED,
    SHARE_DROPPED_LABELS,
    STRATUM_LABELS,
    STUDY,
    UNNORMALISED,
)

CALIBRATION_N = PRIMARY_N
CALIBRATION_REPLICATES = 6_400
NECESSITY_REPLICATES = 2_800
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.5

#: The two construction controls, each a below-band SE control of ``SE_CONTROL_SIDES``.
OWN_CONTROLS = (UNNORMALISED, SHARE_DROPPED)

#: One calibration draw label per setting; each draw writes the rows of all 12 parameters.
PROPERTY_DRAWS: tuple[base.PropertyDraw, ...] = (
    *(
        base.PropertyDraw("interval_calibration", setting, CALIBRATION_REPLICATES, setting)
        for setting in law.SETTINGS
    ),
    base.PropertyDraw("targeting_necessity", "unweighted", NECESSITY_REPLICATES, "unweighted"),
)


def suffixes(record: StudyRecord) -> dict[tuple[str, str], tuple[Any, ...]]:
    """The rule-L3 suffix of every property draw of ``record``."""
    return base.seed_labels(record, PRIMARY_REPLICATES, PROPERTY_DRAWS)


#: The calibration positive arms to the controls derived from their rows.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS, positive_suffix=POSITIVE)

#: The column every property row carries beside the shared schema.
ROW_COLUMNS = ("solver_warnings",)

_Key = tuple[tuple[str, str, int, int, int, int, int, str]]


def _setting_labels(setting: str) -> tuple[str, ...]:
    return tuple(label for label in CALIBRATION_LABELS if label.startswith(f"{setting}_"))


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _, _ = key[0]
    return law.draw(n, sample_seed)


def _cell_of(key: _Key) -> str:
    """The cell whose leading replicates the systematic prefix reads: the marginal ``ate``."""
    family, setting = key[0][:2]
    if family == "targeting_necessity":
        return f"{family}/{setting}_ate_s0__targeted"
    return f"{family}/{setting}_ate_all__{POSITIVE}"


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
    for label in _setting_labels(setting):
        name = f"{family}/{label}__{POSITIVE}"
        cells.append(name)
        if label in STRATUM_LABELS:
            cells.append(f"{family}/{label}__{UNNORMALISED}")
        if label in SHARE_DROPPED_LABELS:
            cells.append(f"{family}/{label}__{SHARE_DROPPED}")
        cells.extend(DERIVED_CELLS[name])
    return cells


def _positive(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    truth: float,
    estimate: Any,
    warnings: int,
) -> dict[str, Any]:
    replicate, n, requested = where
    row = replicate_row(
        property_name=family,
        cell=cell,
        role="positive",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    row["solver_warnings"] = warnings
    return row


def _control(
    family: str,
    cell: str,
    where: tuple[int, int, int],
    truth: float,
    estimate: float,
    standard_error: float,
    warnings: int,
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
        critical=law.CRITICAL,
    )
    row["solver_warnings"] = warnings
    return row


def calibration_rows(
    fitted: law.Measured, setting: str, where: tuple[int, int, int]
) -> list[dict[str, Any]]:
    """The positive rows and the two construction controls of one calibration fit."""
    family = "interval_calibration"
    truth = law.truths(setting)
    rows: list[dict[str, Any]] = []
    for label in _setting_labels(setting):
        parameter_label = label.removeprefix(f"{setting}_")
        name = law.parameter(parameter_label)
        estimate = fitted.result.estimates[name]
        rows.append(
            _positive(family, f"{label}__{POSITIVE}", where, truth[name], estimate, fitted.warnings)
        )
        if label in STRATUM_LABELS:
            rows.append(
                _control(
                    family,
                    f"{label}__{UNNORMALISED}",
                    where,
                    truth[name],
                    float(estimate.psi),
                    law.unnormalised_se(fitted.result, parameter_label),
                    fitted.warnings,
                )
            )
        if label in SHARE_DROPPED_LABELS:
            rows.append(
                _control(
                    family,
                    f"{label}__{SHARE_DROPPED}",
                    where,
                    truth[name],
                    float(estimate.psi),
                    law.share_dropped_se(fitted.result, parameter_label),
                    fitted.warnings,
                )
            )
    return rows


def necessity_rows(fitted: law.Measured, where: tuple[int, int, int]) -> list[dict[str, Any]]:
    """The targeted and untargeted rows of one targeting-necessity fit."""
    family = "targeting_necessity"
    truth = law.truths("unweighted")
    rows: list[dict[str, Any]] = []
    for label in NECESSITY_LABELS:
        parameter_label = label.removeprefix("unweighted_")
        name = law.parameter(parameter_label)
        estimate = fitted.result.estimates[name]
        rows.append(
            _positive(family, f"{label}__targeted", where, truth[name], estimate, fitted.warnings)
        )
        rows.append(
            _control(
                family,
                f"{label}__untargeted",
                where,
                truth[name],
                law.untargeted(fitted.result, parameter_label),
                float(estimate.std_error),
                fitted.warnings,
            )
        )
    return rows


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, setting, replicate, n, requested, _, fold_seed, _ = key[0]
    where = (replicate, n, requested)
    if family == "interval_calibration":
        return calibration_rows(law.measured(frame, setting, fold_seed), setting, where)
    if family == "targeting_necessity":
        fitted = law.measured(frame, setting, fold_seed, learner=law.drop_w1_learner())
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
    controls["solver_warnings"] = controls["solver_warnings"].astype(int)
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
    """The verdict of each cell.

    The study has one size, so it publishes no rate rows (``rate_labels=()``).
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("targeting_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=POSITIVE)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    return finish(summary, rates)


PROPERTY_RULES = (*property_verdicts.necessity_rules(TARGETING_NECESSITY),)
