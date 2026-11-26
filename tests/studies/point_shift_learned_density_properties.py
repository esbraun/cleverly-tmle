"""Repeated-sampling properties of ``point-shift-learned-density`` (X121 part b2).

One family, ``interval_calibration``: the cross-fitted learned-density arm's three contrasts
(:data:`~tests.studies.point_shift_learned_density.CALIBRATION_LABELS`), each read by the shared
calibration rule (:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) with no
efficiency band, beside its shrunken-SE control (factor :data:`SHRUNKEN_SE_FACTOR`) and its noise
control (one efficient-influence-curve SD), both derived from its own rows
(:func:`~tests.studies.evidence.property_verdicts.calibration_controls`).

A property draw's sample seed is ``stream_seed(record, "property_sample", family, label,
replicate)`` and its fit seed the same tuple with ``"fit_seed"`` first. A replication that raises
is not redrawn; the failure accounting applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import _point_shift_learned_density_law as law
from tests.studies import mtp_study_tools as tools
from tests.studies import point_shift_learned_density as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS
CALIBRATION_N = study.PRIMARY_N
FAMILY = "interval_calibration"
LABEL = "crossfit_learned"

#: The calibration draws: the per-cell predictive rule
#: (:func:`~tests.studies.mtp_study_tools.per_arm_replicates`) at a pilot SE ratio of exactly 1
#: for all three cells, with the pilot's own error
#: (:data:`~tests.studies.point_shift_learned_density_pilot.PILOT_DRAWS`). The three contrasts
#: are distinct parameters, not exchangeable arms, so the per-cell rule applies. Before the
#: declared run this count is set to :func:`calibration_count` of the pilot
#: (``test_point_shift_learned_density_design``).
CALIBRATION_REPLICATES = 4_800
#: The cap at the shared spread constant: the 95th percentile, as a multiple of 400, of the
#: per-cell count over simulated ideal pilots of three cells (:func:`calibration_cap`).
CALIBRATION_REPLICATES_CAP = 19_600
#: The absolute ceiling, fixed before the pilot: the cap at the shared constant. A heavier-tailed
#: pilot can push the cap past it, or leave the rule unable to size the cells; the run then
#: proceeds at the ceiling, may come back red on a finite-sample basis, and such a red is routed
#: as the study's reading routes a red with no defect.
CALIBRATION_REPLICATES_CEILING = CALIBRATION_REPLICATES_CAP
#: The simulated pilots behind a cap.
CAP_SIMULATIONS = 1_000


def calibration_cap(c: float) -> int:
    """The 95th percentile, as a multiple of 400, of the per-cell count over ideal pilots at ``c``.

    An ideal pilot draws each cell's ratio as ``1 + N(0, sigma_P^2)`` with
    ``sigma_P = c / sqrt(2 P)``; a pilot the rule cannot size counts as above every cap.
    """
    from tests.studies import point_shift_learned_density_pilot as pilot

    rng = np.random.default_rng(pilot.PILOT_SEED)
    sigma = tools.pilot_sigma(pilot.PILOT_DRAWS, c)
    counts = []
    for _ in range(CAP_SIMULATIONS):
        ratios = (1.0 + sigma * rng.normal(size=len(study.CALIBRATION_LABELS))).tolist()
        sized = tools.per_arm_replicates(ratios, pilot.PILOT_DRAWS, c)["R"]
        counts.append(sized if sized is not None else 10**9)
    return int(np.ceil(np.percentile(counts, 95) / tools.R_STEP) * tools.R_STEP)


def calibration_count(se_ratios: list[float], implied_c: list[float]) -> dict[str, Any]:
    """The declared count from a pilot: the per-cell rule at ``max(C_FIT, pilot c)``, capped.

    Returns the constant used, the rule's count (``None`` when it cannot size the cells), the cap
    at that constant, and the count the run proceeds at: the smallest of the rule, the cap and
    :data:`CALIBRATION_REPLICATES_CEILING`.
    """
    from tests.studies import point_shift_learned_density_pilot as pilot

    c = max(tools.C_FIT, *implied_c)
    rule = tools.per_arm_replicates(se_ratios, pilot.PILOT_DRAWS, c)["R"]
    cap = CALIBRATION_REPLICATES_CAP if c == tools.C_FIT else calibration_cap(c)
    ceiling = CALIBRATION_REPLICATES_CEILING
    declared = min(ceiling if rule is None else rule, cap, ceiling)
    return {"c": c, "rule": rule, "cap": cap, "declared": declared}


#: The shrunken-SE control: the reported SE times 0.70, the shared factor.
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

FAILURE_BOUND = study.FAILURE_BOUND
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The positive arm of each label to the controls derived from it.
DERIVED_CELLS = calibration_derived(study.CALIBRATION_LABELS, positive_suffix=study.POSITIVE)

ROW_COLUMNS = study.HARNESS_COLUMNS

_Key = tuple[tuple[str, str, int, int, int, int, int]]


def _draw(key: _Key) -> pd.DataFrame:
    _, _, _, n, _, sample_seed, _ = key[0]
    return law.draw(n, sample_seed)


def _cells_of(key: _Key) -> list[str]:
    """Every ``property/cell`` one draw writes, derived controls included."""
    del key
    return [f"{FAMILY}/{cell}" for cell in PROPERTY_CELLS[FAMILY]]


def _replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, _, replicate, n, requested, _, fit_seed = key[0]
    result = law.fit(study.GATED_ARM, frame, fit_seed)
    rows = []
    for tag, name in law.CONTRASTS.items():
        row = replicate_row(
            property_name=FAMILY,
            cell=f"{study.GATED_ARM}_{tag}__{study.POSITIVE}",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=law.TRUTH[tag],
            estimate=result.estimates[name],
            alpha=STUDY.margins.alpha,
        )
        diagnostics = law.ratio_diagnostics(result, tag, frame)
        row.update(
            ratio_sd=diagnostics.ratio_sd,
            ratio_mean=diagnostics.ratio_mean,
            ratio_rmse=diagnostics.ratio_rmse,
        )
        rows.append(row)
    return rows


def _payloads(record: StudyRecord, replicates: int) -> list[_Key]:
    out = []
    for replicate in range(replicates):
        sample_seed, fit_seed = (
            stream_seed(record, stream, FAMILY, LABEL, replicate)
            for stream in ("property_sample", "fit_seed")
        )
        out.append(((FAMILY, LABEL, replicate, CALIBRATION_N, replicates, sample_seed, fit_seed),))
    return out


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY, cap: int | None = None
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication.

    ``record`` and ``cap`` exist for a smoke on a throwaway record; the declared run passes
    neither.
    """
    replicates = CALIBRATION_REPLICATES if cap is None else min(cap, CALIBRATION_REPLICATES)
    outcomes = fit_properties(
        _payloads(record, replicates),
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
        labels=study.CALIBRATION_LABELS,
        efficiency_bounds=study.EFFICIENCY_BOUNDS,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
        positive_suffix=study.POSITIVE,
    )
    rows = pd.concat([rows, controls], ignore_index=True)
    cells = set(rows.loc[rows["property"] == FAMILY, "cell"])
    if cells != set(PROPERTY_CELLS[FAMILY]):
        raise RuntimeError(f"{FAMILY} cells {sorted(cells ^ set(PROPERTY_CELLS[FAMILY]))} differ")
    return rows.loc[:, [*REPLICATE_COLUMNS, *ROW_COLUMNS]]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration verdict of each cell. The study has one size, so no rate rows."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=(), rate_labels=())
    calibration_verdicts(
        summary, margins=STUDY.margins, efficiency_band=None, positive_suffix=study.POSITIVE
    )
    return finish(summary, rates)


#: The declared rule of the failure accounting: the shared calibration rule, no efficiency band.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)


__all__ = [
    "CALIBRATION_REPLICATES",
    "DERIVED_CELLS",
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "generate_property_rows",
    "summarize_properties",
]
