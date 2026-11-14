"""Property families of ``option-stopping-ltmle`` (Fix-8 plan section 5.2), and their verdicts.

The cells are :data:`tests.studies.canonical_option_stopping_ltmle.CELLS`.  One draw of each
stream (``L3`` and ``L4``) is fitted once per declared fit, and every cell of the stream reads
those fits, so K, W, N and the reported rows of one law read the same draws.

**Rules.**

- ``interval_calibration`` (K) reads the shared calibration rule
  (:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) on the widened
  interval, with the derived ``shrunken_se_control`` and ``noise_control``
  (:func:`~tests.studies.evidence.property_verdicts.calibration_controls`).  The noise arm adds
  one unit of :data:`NOISE_SD` (the pilot's empirical SD of the cell's estimates, times
  ``sqrt(n)``); the study claims no efficiency band.
- ``residual_witness`` (W) reads the exact interval of the share of draws whose residual is at
  least 0.01 SE (:func:`~tests.studies.evidence.property_verdicts.coverage_limit_verdicts`):
  a gated cell's lower limit is at least
  :data:`~tests.studies.canonical_option_stopping_ltmle.WITNESS_SHARE`.
- ``residual_necessity`` (N) reads the same function on the coverage of one set of fits: the
  plain interval's upper limit strictly below
  :data:`~tests.studies.canonical_option_stopping_ltmle.NECESSITY_CEILING` (control) and the
  widened interval's lower limit at least
  :data:`~tests.studies.canonical_option_stopping_ltmle.NECESSITY_FLOOR` (positive).
- Every other family is reported
  (:func:`~tests.studies.evidence.property_verdicts.reported_verdicts`).

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import _option_stopping_cells as grid
from tests.studies import canonical_option_stopping_ltmle as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    coverage_limit_rule,
    coverage_limit_verdicts,
    finish,
    reported_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
PROPERTY_CELLS = study.PROPERTY_CELLS
CALIBRATION_N = study.PRIMARY_N
REPLICATES = study.REPLICATES
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The pilot's empirical SD of each K cell's estimates times ``sqrt(n)``
#: (``tests/studies/option_stopping_pilot.csv``, 2,000 draws), which sizes the noise arm.
#: The design test checks each against the committed pilot.
NOISE_SD: dict[str, float] = {
    "l3_sv_newton": 1.4132,
    "l3_ss_newton": 1.4132,
    "l4_sv_one_step": 1.0857,
    "l4_ss_one_step": 1.0871,
}

#: The calibration positive arm of each label to the controls derived from it.
DERIVED_CELLS = calibration_derived(study.CALIBRATION_LABELS, positive_suffix=grid.POSITIVE_SUFFIX)

#: The extra columns of a synthetic row of the property differential.
synthetic_columns = grid.synthetic_columns

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

_Payload = tuple[str, int, int, int]
_Key = tuple[_Payload]


def _seed(stream: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", stream, replicate)


def _draw(key: _Key) -> tuple[pd.DataFrame, dict[str, float]]:
    """The sample a key fits, a function of the key alone."""
    stream, replicate, n, _ = key[0]
    return study.STREAMS[stream].draw(n, _seed(stream, replicate))


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names one key's rows and derived controls fill."""
    return grid.stream_cells(study.CELLS, key[0][0])


def cell_of(key: _Key) -> str:
    return f"stream/{key[0][0]}"


def replicate_of(key: _Key) -> int:
    return key[0][1]


def fit_drawn(
    payload: _Payload, drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    """Every row of one key on its drawn sample."""
    stream, replicate, n, requested = payload
    frame, truth = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return grid.stream_rows(
            study.STREAMS[stream],
            study.CELLS,
            frame,
            truth,
            replicate=replicate,
            n=n,
            requested=requested,
            alpha=STUDY.margins.alpha,
        )


def _fit_key(key: _Key, drawn: tuple[pd.DataFrame, dict[str, float]]) -> list[dict[str, Any]]:
    return fit_drawn(key[0], drawn)


def _payloads(budget: int | None, start: int = 0) -> list[_Key]:
    requested = REPLICATES if budget is None else budget
    return [
        ((stream, replicate, CALIBRATION_N, requested),)
        for stream in study.STREAMS
        for replicate in range(start, start + requested)
    ]


def declared_cells() -> tuple[Any, ...]:
    """Every sampled cell, with its role, estimand, size and replicate count."""
    return grid.declared_cells(
        STUDY, study.CELLS, study.truths(), n=CALIBRATION_N, replicates=REPLICATES
    )


def with_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """The fitted rows and the calibration controls derived from them, in published order."""
    controls = calibration_controls(
        rows,
        STUDY,
        labels=study.CALIBRATION_LABELS,
        efficiency_bounds=NOISE_SD,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
        positive_suffix=grid.POSITIVE_SUFFIX,
    )
    return grid.published(pd.concat([rows, controls], ignore_index=True))


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each stream for a pre-run check."""
    outcomes = fit_properties(
        _payloads(budget),
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
        seed_of=lambda key: _seed(key[0][0], key[0][1]),
    )
    rows = with_controls(pd.DataFrame([row for result in outcomes for row in result]))
    grid.check_cells(rows, PROPERTY_CELLS)
    return rows


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts of K, W and N; every other family reported."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=grid.SUMMARY_COLUMNS, rate_labels=()
    )
    calibration_verdicts(
        summary, margins=STUDY.margins, efficiency_band=None, positive_suffix=grid.POSITIVE_SUFFIX
    )
    coverage_limit_verdicts(summary, family="residual_witness", floor=study.WITNESS_SHARE)
    coverage_limit_verdicts(
        summary,
        family="residual_necessity",
        floor=study.NECESSITY_FLOOR,
        ceiling=study.NECESSITY_CEILING,
    )
    reported_verdicts(summary, families=study.REPORTED_FAMILIES)
    grid.rule_columns(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rules.  K reads the shared calibration rule
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (
    coverage_limit_rule("residual_witness", floor=study.WITNESS_SHARE, ceiling=None),
    coverage_limit_rule(
        "residual_necessity", floor=study.NECESSITY_FLOOR, ceiling=study.NECESSITY_CEILING
    ),
)


def smoke_keys(draws: int, start: int) -> list[_Key]:
    """Keys of a smoke on replicate ids ``start`` to ``start + draws - 1`` of every stream."""
    return _payloads(draws, start)


def fit_keys(keys: Sequence[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_fit_key(key, _draw(key)) for key in keys]


__all__ = [
    "DERIVED_CELLS",
    "FAILURE_BOUND",
    "NOISE_SD",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
]
