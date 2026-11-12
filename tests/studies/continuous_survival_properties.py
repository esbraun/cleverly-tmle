r"""Repeated-sampling properties of the continuous-time survival TMLE, and its reported cell.

Two families, fixed before any run, cross-fitted with the study's 10 folds
(:mod:`tests.studies.continuous_survival`).  "Wrong" is :func:`~tests.studies.continuous_survival.engine`'s:
the event hazards from the Cox model on the treatment alone (it ignores every confounder), the
treatment model the unpenalized logistic on
:math:`X_1, X_2`, and the censoring hazard the Cox model on the treatment alone.

**Double robustness** (:data:`ROBUSTNESS_REPLICATES` each, :math:`n = 1{,}000`, on the
dependent-censoring law, so a wrong censoring model is wrong).

=====================================  ========  ============================================
cell                                   role      configuration
=====================================  ========  ============================================
``treated_c1__mechanism_correct``      positive  event hazards wrong; treatment and censoring
                                                 correct
``treated_c1__outcome_correct``        positive  event hazards correct; treatment and
                                                 censoring wrong
``treated_c1__both_wrong``             control   all three wrong
=====================================  ========  ============================================

Each reads the treated arm's risk of cause 1 at the middle target time, the risk the
arm-specific term moves.  The control's gap is read from the pilot and stated in
``tests/canonical/continuous_survival/regenerate.py`` before the run.

**Calibration** (:data:`CALIBRATION_REPLICATES`, :math:`n = 1{,}000`).  The positive cell
``both_rd__correctly_specified`` fits the record's ``both_cf_1k`` configuration on the base law
and reads cause 1's risk difference at the middle time; its SE ratio is read against the shared
calibration band and its efficiency ratio against :data:`EFFICIENCY_RATIO_BAND`, with the
efficient spread of :func:`~tests.studies.continuous_survival_law.efficiency_sd`.  The two
shared controls are derived from its rows.

**Reported cells** (no verdict; each cell's count in :data:`REPORTED_CELLS`, on the stream
``stream_seed(STUDY, "reported", cell, r)``).  ``both_fs_1k``, ``both_fs_4k`` and ``both_cf_4k``
fit the default libraries on the base law in the settings the record does not gate.
``positivity_4k`` fits them on Hage's S2 treatment law, :math:`\omega = (6, -1, 1)`, where a
third of the units have a treatment probability below 0.025 or above 0.975; it is expected to
under-cover.  All four publish as ``reported-cells.csv`` and ``reported-replicates.csv``.

A replication that raises is never redrawn.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import recordable
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as study
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY

ROBUSTNESS = "double_robustness"
CALIBRATION = "interval_calibration"

#: The property cells run at the record's gated size (Eric's decision, 2026-11-12: at 4,000
#: the shared rules asked for about 800 core-hours).
ROBUSTNESS_N = 1_000
#: Provisional until the n = 1,000 pilot is read.
ROBUSTNESS_REPLICATES = 1_000
CALIBRATION_N = 1_000
#: Provisional until the n = 1,000 pilot is read.
CALIBRATION_REPLICATES = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

MIDDLE = law.TIMES[1]
TREATED_C1 = law.level_name("treated", "cause1", MIDDLE)
RD = law.difference_name("cause1", MIDDLE)

#: ``cell -> (configuration, role)``.
ROBUSTNESS_CELLS: dict[str, tuple[str, str]] = {
    "treated_c1__mechanism_correct": ("mechanism_correct", "positive"),
    "treated_c1__outcome_correct": ("outcome_correct", "positive"),
    "treated_c1__both_wrong": ("both_wrong", "control"),
}
CALIBRATION_LABELS = ("both_rd",)
EFFICIENCY_SD = {"both_rd": law.efficiency_sd(1, MIDDLE)["difference"]}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

#: ``cell -> (configuration, n, replicates)`` of the reported cells: the default fit in the
#: settings the record does not gate, and the positivity law.
REPORTED_CELLS: dict[str, tuple[str, int, int]] = {
    "both_fs_1k": ("default_fs", 1_000, 1_000),
    "both_fs_4k": ("default_fs", 4_000, 400),
    "both_cf_4k": ("default_cf", 4_000, 400),
    "positivity_4k": ("positivity", 4_000, 400),
}
CELLS_FILE = "reported-cells.csv"
REPLICATES_FILE = "reported-replicates.csv"


# ---------------------------------------------------------------------- the fits

#: ``configuration -> (law, engine settings)``.
CONFIGURATIONS: dict[str, tuple[str, dict[str, bool]]] = {
    "mechanism_correct": ("dependent_censoring", {"hazard_wrong": True}),
    "outcome_correct": (
        "dependent_censoring",
        {"propensity_wrong": True, "censoring_wrong": True},
    ),
    "both_wrong": (
        "dependent_censoring",
        {"hazard_wrong": True, "propensity_wrong": True, "censoring_wrong": True},
    ),
    "positivity": ("positivity", {}),
    "default_fs": ("base", {"cross_fit": False}),
    "default_cf": ("base", {}),
}


def configured_fit(frame: pd.DataFrame, configuration: str) -> Any:
    """The subject's fit of one property or reported configuration."""
    _, settings = CONFIGURATIONS[configuration]
    return study.fit_with(frame, study.engine(**settings))


# ---------------------------------------------------------------------------- keys

#: ``(family, label, replicate, requested)``: a robustness cell, or a calibration label.
_Key = tuple[str, str, int, int]


def _keys(budget: int | None) -> list[_Key]:
    robust = ROBUSTNESS_REPLICATES if budget is None else min(budget, ROBUSTNESS_REPLICATES)
    calibrate = CALIBRATION_REPLICATES if budget is None else min(budget, CALIBRATION_REPLICATES)
    return [
        *((ROBUSTNESS, cell, r, robust) for cell in ROBUSTNESS_CELLS for r in range(robust)),
        *(
            (CALIBRATION, label, r, calibrate)
            for label in CALIBRATION_LABELS
            for r in range(calibrate)
        ),
    ]


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, _ = key
    seed = stream_seed(STUDY, "property_sample", family, label, replicate)
    if family == ROBUSTNESS:
        configuration, _ = ROBUSTNESS_CELLS[label]
        law_name, _ = CONFIGURATIONS[configuration]
        return study.sample(ROBUSTNESS_N, seed, law_name=law_name)
    return study.sample(CALIBRATION_N, seed)


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, requested = key
    alpha = STUDY.margins.alpha
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == ROBUSTNESS:
            configuration, role = ROBUSTNESS_CELLS[label]
            estimate = configured_fit(frame, configuration)[TREATED_C1]
            return [
                replicate_row(
                    property_name=ROBUSTNESS,
                    cell=label,
                    role=role,
                    replicate=replicate,
                    n=ROBUSTNESS_N,
                    requested=requested,
                    truth=law.TRUTH[TREATED_C1],
                    estimate=estimate,
                    alpha=alpha,
                )
            ]
        estimate = study.fit(frame, study.GATED_CELL)[RD]
    return [
        replicate_row(
            property_name=CALIBRATION,
            cell=f"{label}__correctly_specified",
            role="positive",
            replicate=replicate,
            n=CALIBRATION_N,
            requested=requested,
            truth=law.TRUTH[RD],
            estimate=estimate,
            alpha=alpha,
        )
    ]


def _cells_of(key: _Key) -> list[str]:
    family, label, _, _ = key
    if family == ROBUSTNESS:
        return [f"{ROBUSTNESS}/{label}"]
    name = f"{CALIBRATION}/{label}__correctly_specified"
    return [name, *DERIVED_CELLS[name]]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries."""
    robust = tuple(
        (ROBUSTNESS, cell, TREATED_C1, law.TRUTH[TREATED_C1]) for cell in ROBUSTNESS_CELLS
    )
    calibrated = tuple(
        (CALIBRATION, f"{label}__{kind}", RD, law.TRUTH[RD])
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    )
    return robust + calibrated


#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a smoke run."""
    outcomes = fit_properties(
        _keys(budget),
        _fit_key,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CALIBRATION_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    if budget is not None:
        # A capped smoke run requests what it runs, so the summary reads it as complete.
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts: the union-model rule and the calibration rule with controls."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The union-model family reads the shared rule.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),)


# ------------------------------------------------------------------- reported cells


def reported_seed(cell: str, replicate: int) -> int:
    """The seed of replicate ``replicate`` of the reported cell ``cell``."""
    return stream_seed(STUDY, "reported", cell, replicate)


def _reported(cell: str, replicate: int) -> list[dict[str, Any]]:
    configuration, n, _ = REPORTED_CELLS[cell]
    law_name, _ = CONFIGURATIONS[configuration]
    frame = study.sample(n, reported_seed(cell, replicate), law_name=law_name)
    base = {"scenario": cell, "replicate": replicate, "n": n}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = configured_fit(frame, configuration)
    except Exception as error:
        if not recordable(error):
            raise
        return [
            {
                **base,
                "estimand": name,
                "truth": truth,
                "estimate": np.nan,
                "std_error": np.nan,
                "covered": np.nan,
                "failure": type(error).__name__,
            }
            for name, truth in law.TRUTH.items()
        ]
    rows = []
    for name, truth in law.TRUTH.items():
        estimate = result[name]
        low, high = estimate.ci
        rows.append(
            {
                **base,
                "estimand": name,
                "truth": truth,
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "covered": float(low <= truth <= high),
                "failure": "",
            }
        )
    return rows


def reported_rows(*, budget: int | None = None, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Every reported cell's replicate rows, in a fixed order."""
    jobs = [
        (cell, r)
        for cell, (_, _, count) in REPORTED_CELLS.items()
        for r in range(count if budget is None else min(budget, count))
    ]
    outcomes = map_parallel(_reported, jobs, n_jobs=n_jobs)
    rows = pd.DataFrame([row for rows in outcomes for row in rows])
    return rows.sort_values(["scenario", "replicate", "estimand"], ignore_index=True)


def summarize_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell and estimand: fits, failures, bias, spread, standardized bias, SE, coverage."""
    records = []
    for (cell, estimand), group in rows.groupby(["scenario", "estimand"], sort=True):
        kept = group.loc[group["failure"].fillna("").eq("")]
        error = kept["estimate"] - kept["truth"]
        spread = float(error.std(ddof=1))
        records.append(
            {
                "scenario": cell,
                "estimand": estimand,
                "n": int(group["n"].iloc[0]),
                "fits": len(kept),
                "failures": len(group) - len(kept),
                "bias": float(error.mean()),
                "empirical_sd": spread,
                "standardized_bias": float(error.mean() / spread),
                "mean_std_error": float(kept["std_error"].mean()),
                "se_ratio": float(kept["std_error"].mean() / spread),
                "coverage": float(kept["covered"].mean()),
            }
        )
    return pd.DataFrame(records)


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The property rows and the reported cell's two artifacts."""
    rows = reported_rows(budget=budget, n_jobs=n_jobs)
    return (
        generate_property_rows(n_jobs=n_jobs, budget=budget),
        {CELLS_FILE: summarize_cells(rows), REPLICATES_FILE: rows},
    )
