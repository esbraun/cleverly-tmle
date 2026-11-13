"""The ``band-t-reference`` cells: the ``t`` band and its paired controls on each fit shape.

One family, ``simultaneous_coverage``.  Each cell of :data:`tests.studies.band_t_reference.CELLS`
publishes, from one fit per replicate:

=========================================  =================  ================================
cell                                       role               reading
=========================================  =================  ================================
``<label>__t_band``                        positive           the joint rule: the 99% CP
                                                              interval inside (0.92, 0.98)
``<label>__normal_reference_control``      reference_control  the default band on the same
                                                              fit; the paired gain, reported
``<label>__pointwise_joint_control``       control            joint pointwise coverage below
                                                              0.95
``<label>__<arm>_diagnostic``              diagnostic         a secondary df rule, reported
=========================================  =================  ================================

:func:`tests.studies.evidence.simultaneous.t_band_rows` builds every row.  The fit runs once
with the default band; the ``t`` band is rebuilt from the same estimates, draws and ``c``.  A
replicate whose band family holds a constant arm cell (:func:`constant_arm_cells`, this
module's copy of Fix-3's check) is refused and counted not covered on every band row.

Seeds: ``stream_seed(STUDY, "property_sample", "simultaneous_coverage", label, r)`` for the
run, and the salted ``band-t-reference/pilot`` stream for the pilot, so no draw is shared with
``default-simultaneous-bands``, ``canonical-stratified-tmle`` or ``longitudinal-sdr``.

The reference control's paired gain is a reported reading (plan rev 3.3): the summary carries
its interval and the row passes; :func:`reference_readings` sets it beside the declared value.
It is not in :data:`PROPERTY_RULES` and sizes nothing.  The joint rule, the pointwise control
and the deterministic checks inside ``t_band_rows`` stay gated.

``python -m tests.studies.band_t_reference_properties --smoke`` fits
:data:`~tests.studies.band_t_reference.SMOKE_LABELS` at
:data:`~tests.studies.band_t_reference.SMOKE_REPLICATES` replicates on the pilot stream and
prints the per-cell coverage; it writes nothing.  ``--pilot`` runs the declared pilot and writes
:data:`~tests.studies.band_t_reference.PILOT_PATH`.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import band_t_reference as study
from tests.studies.band_t_reference import CELL_BY_LABEL, DECLARATION, STUDY, Cell, Declaration
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties, recording
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import (
    FAMILY,
    SCALE_GAIN_FLOOR,
    t_band_rows,
    t_property_cells,
)

CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The per-replicate diagnostics file (plan 5.2), written beside the rows.
DIAGNOSTICS_FILE = "t-band-replicates.csv"

Sample = tuple[pd.DataFrame, dict[str, float]]


@dataclass(frozen=True)
class Shape:
    """How one fit shape draws, fits and reads its constant arm cells.

    Parameters
    ----------
    draw : callable
        ``(n, seed) -> (frame, truth)``.
    fit : callable
        ``frame -> result`` with the default band.
    arms : callable
        ``frame -> tuple of str``: the shape's constant arm cells, empty where Fix-3 sets
        no status.
    """

    draw: Callable[[int, int], Sample]
    fit: Callable[[pd.DataFrame], Any]
    arms: Callable[[pd.DataFrame], tuple[str, ...]]


def _constant(
    frame: pd.DataFrame,
    by: Sequence[str],
    *,
    levels: pd.DataFrame | None = None,
    outcome: str = "Y",
) -> tuple[str, ...]:
    """Each ``by`` cell whose observed outcomes take one value or none, as ``"V=2|A=1"``.

    The cells are the full product of each column's levels in ``levels`` (``frame`` itself by
    default), so a cell with no rows is constant too.
    """
    source = frame if levels is None else levels
    index = pd.MultiIndex.from_product(
        [sorted(source[name].dropna().unique()) for name in by], names=list(by)
    )
    counts = (
        frame.dropna(subset=[outcome])
        .groupby(list(by))[outcome]
        .nunique()
        .reindex(index, fill_value=0)
    )
    return tuple(
        "|".join(f"{name}={value:g}" for name, value in zip(by, key, strict=True))
        for key, count in counts.items()
        if count <= 1
    )


def constant_arm_cells(source: str, frame: pd.DataFrame) -> tuple[str, ...]:
    """This study's copy of Fix-3's constant-arm-cell check, shape by shape (plan 5.1).

    On the point-treatment shapes an arm, within the stratum where the shape has strata, whose
    observed outcomes over rows with positive weight take one value (or none: an empty cell of
    the stratum-by-arm product counts).  Every family these shapes band holds a contrast of both
    arms in each stratum it reports, so one such cell puts a constant-cell member in the band,
    and the replicate is refused.  On ``cde_z0`` the rows with weight are those with ``Z = 0``
    and an observed outcome.  On the longitudinal shapes Fix-3 sets no status, and the copy is
    empty.  The design test asserts it agrees with ``constant_outcome_cells`` on every
    non-empty cell.
    """
    if source in {"strata", "crossfit_strata"}:
        return _constant(frame, ("V", "A"))
    if source in {"ordinary_binary", "point_msm"}:
        return _constant(frame, ("A",))
    if source == "cde_z0":
        kept = frame.loc[(frame["Z"] == 0) & (frame["Delta"] == 1)]
        return _constant(kept, ("A",), levels=frame)
    if source in {"categorical_ltmle", "categorical_ltmle_crossfit", "six_regimens"}:
        return ()
    raise KeyError(source)


def covariate_cells(frame: pd.DataFrame) -> np.ndarray:
    """Each row's covariate cell for the model-assisted df: every column but the outcome."""
    keys = [column for column in frame.columns if column != "Y"]
    codes = np.column_stack(pd.MultiIndex.from_frame(frame[keys]).codes)
    return np.unique(codes, axis=0, return_inverse=True)[1].reshape(-1)


def _default_source(label: str) -> Shape:
    from tests.studies.default_band_properties import SOURCES

    source = SOURCES[label]

    def draw(n: int, seed: int) -> Sample:
        return source.draw(n, seed)

    def fit(frame: pd.DataFrame) -> Any:
        return source.fit(frame)

    def arms(frame: pd.DataFrame) -> tuple[str, ...]:
        return constant_arm_cells(label, frame)

    return Shape(draw=draw, fit=fit, arms=arms)


def _strata(crossfit: bool) -> Shape:
    from tests.studies import stratified_law as law
    from tests.studies.canonical_stratified_tmle import fit_cleverly
    from tests.studies.stratified_tmle_properties import CALIBRATION_ESTIMANDS, CROSSFIT_ESTIMANDS

    def draw(n: int, seed: int) -> Sample:
        return law.sample(n, seed), {name: float(value) for name, value in law.TRUTH.items()}

    def fit(frame: pd.DataFrame) -> Any:
        return fit_cleverly(
            frame,
            estimands=CROSSFIT_ESTIMANDS if crossfit else CALIBRATION_ESTIMANDS,
            cross_fit=crossfit,
            simultaneous=True,
        )

    label = "crossfit_strata" if crossfit else "strata"
    return Shape(draw=draw, fit=fit, arms=lambda frame: constant_arm_cells(label, frame))


def _six_regimens() -> Shape:
    from tests import discrete_law_longitudinal as law
    from tests.studies import longitudinal_sdr_properties as sdr
    from tests.studies import ltmle_crossfit_properties as end_properties

    def draw(n: int, seed: int) -> Sample:
        frame = end_properties.sample(law.PROBS, n, seed)
        return frame, {name: float(law.TRUTH[name]) for name in sdr.BAND_NAMES}

    def fit(frame: pd.DataFrame) -> Any:
        return sdr.discrete_fit(
            frame, regimens=sdr.BAND_REGIMENS, simultaneous=True, band=sdr.BAND_OPTIONS
        )

    return Shape(draw=draw, fit=fit, arms=lambda frame: ())


def shape(source: str) -> Shape:
    """The fit shape a cell's ``source`` names."""
    if source in {"strata", "crossfit_strata"}:
        return _strata(source == "crossfit_strata")
    if source == "six_regimens":
        return _six_regimens()
    return _default_source(source)


def fit_cell(
    cell: Cell, frame: pd.DataFrame, truth: dict[str, float]
) -> tuple[Any, dict[str, float]]:
    """The cell's fit and the truth of every name it reports."""
    result = shape(cell.source).fit(frame)
    missing = [name for name in result.estimates if name not in truth]
    if missing:
        raise KeyError(f"{cell.label} has no truth for {missing}")
    return result, {name: float(truth[name]) for name in result.estimates}


_Key = tuple[tuple[str, int, int, int]]


def _draw(key: _Key) -> Sample:
    label, _, _, sample_seed = key[0]
    cell = CELL_BY_LABEL[label]
    return shape(cell.source).draw(cell.n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    return [f"{FAMILY}/{name}" for name in study.cell_names(CELL_BY_LABEL[key[0][0]])]


def _cell_of(key: _Key) -> str:
    return f"{FAMILY}/{study.BAND_CELLS[key[0][0]]}"


def _replicate_of(key: _Key) -> int:
    return key[0][1]


def fit_replication(key: _Key, drawn: Sample) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One replicate's rows and its diagnostics record."""
    label, replicate, requested, _ = key[0]
    cell = CELL_BY_LABEL[label]
    frame, truth = drawn
    result, full = fit_cell(cell, frame, truth)
    started = time.perf_counter()
    rows, record = t_band_rows(
        result,
        full,
        tuple(result.estimates),
        label=label,
        replicate=replicate,
        n=cell.n,
        requested=requested,
        pointwise_critical=CRITICAL,
        constant_cells=shape(cell.source).arms(frame),
        arms=cell.arms,
        cells=covariate_cells(frame),
    )
    record["read_seconds"] = time.perf_counter() - started
    return rows, record


#: The share of a cell's requested replications the accounting lets a run lose.
FAILURE_BOUND = 0.01
#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def seed(
    label: str, replicate: int, *, pilot: bool = False, declaration: Declaration = DECLARATION
) -> int:
    """The sample seed of one replicate: the study's run stream, or its salted pilot stream."""
    stream = declaration.pilot_stream if pilot else "property_sample"
    return stream_seed(declaration.record, stream, FAMILY, label, replicate)


def payloads(
    labels: Sequence[str] | None = None,
    budget: int | None = None,
    *,
    pilot: bool = False,
    declaration: Declaration = DECLARATION,
) -> list[_Key]:
    """Every ``(label, replicate, requested, seed)`` key of a study, in cell order."""
    chosen = (
        declaration.cells if labels is None else tuple(CELL_BY_LABEL[label] for label in labels)
    )
    total = study.PILOT_REPLICATES if pilot else declaration.replicates
    count = total if budget is None else min(budget, total)
    return [
        (
            (
                cell.label,
                replicate,
                count,
                seed(cell.label, replicate, pilot=pilot, declaration=declaration),
            ),
        )
        for cell in chosen
        for replicate in range(count)
    ]


def _fit_all(
    keys: list[_Key], n_jobs: int, *, checkpoint: Any = "ambient"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = fit_properties(
        keys,
        fit_replication,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        rows_at=0,
        seed_of=lambda key: key[0][3],
        checkpoint=checkpoint,
    )
    rows = pd.DataFrame([row for value in outcomes for row in value[0]])
    records = pd.DataFrame([value[1] for value in outcomes])
    return rows, records


def _ordered(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def property_artifacts(
    declaration: Declaration, *, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every declared replicate of a study; its rows and per-replicate diagnostics."""
    rows, records = _fit_all(payloads(declaration=declaration), n_jobs)
    return _ordered(rows), {DIAGNOSTICS_FILE: records}


def declared_thresholds(declaration: Declaration) -> dict[str, float | None]:
    """Each label's declared gain value, recomputed from the study's committed pilot at its R.

    The design test compares it with the committed ``declaration.reference_thresholds``.
    """
    return study.reference_thresholds(
        study.read_pilot(declaration.pilot_path), declaration.replicates, declaration.cells
    )


def summarize(declaration: Declaration, rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary of a study: the joint rule gated, the paired gains reported.

    Every reference control is read with no threshold, so its row carries the gain interval
    and passes (plan rev 3.3).
    """
    summary, rates = apply_shared_verdicts(rows, declaration.record, rate_labels=())
    simultaneous_coverage_verdicts(
        summary,
        margins=declaration.record.margins,
        rows=rows,
        record=declaration.record,
        reference_thresholds=dict.fromkeys(declaration.reference_thresholds),
    )
    return finish(summary, rates)


def reference_readings(declaration: Declaration, summary: pd.DataFrame) -> pd.DataFrame:
    """Each reference control's gain interval beside its declared value, reported only.

    ``reached`` is whether the 99% lower limit reaches the declared value and the witness
    floor; it is empty where the cell declares no value.  No verdict reads it.
    """
    out = []
    for cell in declaration.cells:
        row = summary.loc[summary["cell"] == study.REFERENCE_CELLS[cell.label]].iloc[0]
        threshold = declaration.reference_thresholds[cell.label]
        low = float(row["coverage_gain_ci_lower"])
        out.append(
            {
                "label": cell.label,
                "group": cell.group,
                "declared_value": threshold,
                "gain_ci_lower": low,
                "gain_ci_upper": float(row["coverage_gain_ci_upper"]),
                "reached": None
                if threshold is None
                else bool(low >= max(threshold, SCALE_GAIN_FLOOR)),
            }
        )
    return pd.DataFrame(out)


def property_cells_of(declaration: Declaration) -> tuple[PropertyCell, ...]:
    """Every declared cell of a study, each naming the zero-truth law its rows read."""
    return tuple(
        item
        for cell in declaration.cells
        for item in t_property_cells(
            cell.label,
            n=cell.n,
            replicates=declaration.replicates,
            seed=seed(cell.label, 0, declaration=declaration),
            arms=cell.arms,
        )
    )


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every declared replicate; the rows, and the per-replicate diagnostics by file."""
    return property_artifacts(DECLARATION, n_jobs=n_jobs)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Fit every declared replicate of every cell."""
    return generate_property_artifacts(n_jobs=n_jobs)[0]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared summary: the joint rule gated, the paired gains reported."""
    return summarize(DECLARATION, rows)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every declared cell, each naming the zero-truth law its rows are read against."""
    return property_cells_of(DECLARATION)


#: The failure accounting's declared rules: the joint rule, without the reported reference gain.
PROPERTY_RULES = (property_verdicts.simultaneous_rule(),)


# ------------------------------------------------------------------------ pilot and smoke


def pilot_readings(
    rows: pd.DataFrame, records: pd.DataFrame, cells: Sequence[Cell] = tuple(CELL_BY_LABEL.values())
) -> pd.DataFrame:
    """Each label's ``t`` coverage, paired gain and SD, refused and constant shares."""
    out = []
    for cell in cells:
        band = rows.loc[rows["cell"] == study.BAND_CELLS[cell.label]].set_index("replicate")
        if band.empty:
            continue
        control = rows.loc[rows["cell"] == study.REFERENCE_CELLS[cell.label]].set_index("replicate")
        gains = (band["covered"] - control.loc[band.index, "covered"]).to_numpy(dtype=float)
        mine = records.loc[records["label"] == cell.label]
        out.append(
            {
                "label": cell.label,
                "replicates": len(band),
                "coverage": float(band["covered"].mean()),
                "normal_coverage": float(control["covered"].mean()),
                "gain": float(gains.mean()),
                "gain_sd": float(gains.std(ddof=1)) if len(gains) > 1 else math.nan,
                "refused_share": float(mine["refused"].mean()),
                "constant_share": float((mine["constant_cells"].astype(str) != "").mean()),
                "median_min_df": float(mine["min_df"].median()),
                "max_relative_widening": float(mine["max_relative_widening"].max()),
            }
        )
    return pd.DataFrame(out)


def smoke(
    declaration: Declaration = DECLARATION,
    labels: Sequence[str] | None = None,
    replicates: int = study.SMOKE_REPLICATES,
) -> pd.DataFrame:
    """Fit a study's smoke cells at ``replicates`` on its pilot stream, in process."""
    chosen = declaration.smoke_labels if labels is None else tuple(labels)
    keys = payloads(chosen, replicates, pilot=True, declaration=declaration)
    started = time.perf_counter()
    with recording(declaration.record.implementation, witness=PROPERTY_WITNESS):
        rows, records = _fit_all(keys, 1, checkpoint=None)
    readings = pilot_readings(rows, records)
    readings["seconds_per_fit"] = (time.perf_counter() - started) / len(keys)
    return readings


def run_pilot(declaration: Declaration = DECLARATION, *, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """The declared pilot (plan 5.1): every cell at 800 on the study's salted stream."""
    with recording(declaration.record.implementation, witness=PROPERTY_WITNESS):
        rows, records = _fit_all(
            payloads(pilot=True, declaration=declaration), n_jobs, checkpoint=None
        )
    readings = pilot_readings(rows, records)
    readings.to_csv(declaration.pilot_path, index=False)
    return readings


def main(argv: Sequence[str] | None = None, declaration: Declaration = DECLARATION) -> None:
    """``--smoke`` or ``--pilot`` for one study."""
    parser = argparse.ArgumentParser(description=f"{declaration.record.slug} pilot and smoke")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--pilot", action="store_true")
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    arguments = parser.parse_args(argv)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        if arguments.smoke:
            print(smoke(declaration))
        else:
            print(run_pilot(declaration, n_jobs=arguments.jobs))


if __name__ == "__main__":
    main()
