r"""Repeated-sampling properties of separable effects on hazard ``LTMLE``.

Two families, fixed before any run.

**Double robustness.**  One cell per claim of the technical reference, each on
``cif_regimen[sep10, relapse @ t=3]`` at n = 16,000.  The design test reads each cell's
exact-law limit.

===============================================  ========  ==================================
cell                                             role      what it witnesses
===============================================  ========  ==================================
``sep10_relapse_t3__event_hazard_wrong``         positive  the event-hazard factor's targeting:
                                                           :math:`h_Y` wrong at the level the
                                                           target reads, :math:`\lambda_D` and
                                                           the mechanism right
``sep10_relapse_t3__competing_hazard_wrong``     positive  the competing factor's targeting:
                                                           :math:`\lambda_D` wrong at the level
                                                           the target reads, :math:`h_Y` and
                                                           the mechanism right
``sep10_relapse_t3__both_hazards_wrong``         control   both factors wrong with the exact
                                                           mechanism: the remainder holds their
                                                           product, so the limit misses
``sep10_relapse_t3__mechanism_wrong``            positive  cross-fitted saturated hazards with
                                                           a treatment model that reads no
                                                           covariate
===============================================  ========  ==================================

The first three run the factor-wise engine (:func:`cleverly.longitudinal.separable.walk_separable`)
on one target with the *exact* treatment and censoring mechanisms of the law and the sample's
cell hazards shifted on the logit scale in one factor or both
(:data:`~tests.studies.separable_effects_law.SHIFT_D`,
:data:`~tests.studies.separable_effects_law.SHIFT_Y`).  ``LTMLE`` takes one hazard learner for
both causes, so a wrong factor alone cannot be declared through it; the engine is the code the
fit runs, and the primary scenarios measure the wiring.  One target keeps the factor's six
``(W, node)`` cells at each level from being pinned by the scores (design R2).  The target reads
:math:`\lambda_D` at level 0 and :math:`h_Y` at level 1 only, so only those entries of the
shifts reach it.  The fourth cell runs the study's cross-fitted ``LTMLE`` fit with
``DummyClassifier(strategy="prior")`` as the treatment learner.  In sample it would witness
nothing: saturated hazards fitted on the same rows already solve every score that depends on
``(W, A, node)``, so the estimate equals the correct-mechanism one whatever the treatment model
(the design test pins that identity).  Out of fold the hazards miss the fold's cell means and
the wrong mechanism changes the targeting.

**Calibration** (n = 16,000).  ``in_sample_sde_relapse_t3`` reads
``ate_regimen[sep10 vs a0, relapse @ t=3]``, the separable direct effect, in sample, and
``cross_fitted_sep01_death_t3`` reads ``cif_regimen[sep01, death @ t=3]`` with two folds.  The
efficiency ratios read the exact EIF spread against :data:`EFFICIENCY_RATIO_BAND`.
:data:`CALIBRATION_REPLICATES` is the pooled rule of :mod:`tests.studies.repeated_sizing` on the
pilot, and :data:`ROBUSTNESS_REPLICATES` the count at which each positive cell passes at its
pilot's 80% upper bias; the design test reads both.

A replication that raises is never redrawn.  A fit that stops short of ``tol`` is kept, as in
the primary scenarios.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from cleverly.inference import make_estimate
from cleverly.longitudinal.separable import (
    SeparableState,
    SeparableTarget,
    influence,
    walk_separable,
)
from tests.parallel import STUDY_JOBS
from tests.studies import separable_effects as study
from tests.studies import separable_effects_law as law
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

ROBUSTNESS_N = 16_000
#: The smallest count at which each positive cell passes at its pilot's 80% upper bias.
ROBUSTNESS_REPLICATES = 2_000
CALIBRATION_N = 16_000
#: The pooled calibration rule on the pilot (design test).
CALIBRATION_REPLICATES = 4_400

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

ROBUSTNESS_ESTIMAND = "cif_regimen[sep10, relapse @ t=3]"
#: The one engine target of the robustness cells.
TARGET = SeparableTarget("sep10", (0, 1), "event", "relapse", 3)
#: The engine's walk settings: the study's step and the shipped tolerance.
STEP_SIZE = 0.05
TOL = 1e-10

#: ``cell -> (shift the death factor, shift the relapse factor, role)``; ``None`` is the
#: ``LTMLE`` fit with a wrong treatment model.
ROBUSTNESS_CELLS: dict[str, tuple[bool, bool, str] | None] = {
    "sep10_relapse_t3__event_hazard_wrong": (False, True, "positive"),
    "sep10_relapse_t3__competing_hazard_wrong": (True, False, "positive"),
    "sep10_relapse_t3__both_hazards_wrong": (True, True, "control"),
    "sep10_relapse_t3__mechanism_wrong": None,
}
ROBUSTNESS_ROLES = {
    cell: ("positive" if spec is None else spec[2]) for cell, spec in ROBUSTNESS_CELLS.items()
}

#: ``label -> (scenario of the fit, estimand)``.
CALIBRATION_CELLS: dict[str, tuple[str, str]] = {
    "in_sample_sde_relapse_t3": (study.IN_SAMPLE, "ate_regimen[sep10 vs a0, relapse @ t=3]"),
    "cross_fitted_sep01_death_t3": (study.CROSS_FITTED, "cif_regimen[sep01, death @ t=3]"),
}
CALIBRATION_LABELS = tuple(CALIBRATION_CELLS)
CALIBRATION_TRUTH = {label: law.TRUTH[name] for label, (_, name) in CALIBRATION_CELLS.items()}
EFFICIENCY_SD = {label: study.EFFICIENCY_SD[name] for label, (_, name) in CALIBRATION_CELLS.items()}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)


class _EngineFit:
    """The engine's targeted estimate of :data:`ROBUSTNESS_ESTIMAND`, read like a result."""

    def __init__(self, estimate: Any, record: Any) -> None:
        self.estimate = estimate
        self.record = record

    def __getitem__(self, name: str) -> Any:
        if name != ROBUSTNESS_ESTIMAND:  # pragma: no cover - guards the cell names
            raise KeyError(name)
        return self.estimate


def engine_fit(frame: pd.DataFrame, shift_d: bool, shift_y: bool) -> _EngineFit:
    """The factor-wise engine on one target, from the sample's cell hazards with the
    declared shifts, at the exact mechanism."""
    inputs, start = law.engine_inputs(frame)
    zero = np.zeros_like(start.beta_d)
    delta_d = law.logit_shift(frame, law.SHIFT_D) if shift_d else zero
    delta_y = law.logit_shift(frame, law.SHIFT_Y) if shift_y else zero
    shifted = SeparableState(start.beta_d + delta_d, start.beta_y + delta_y)
    targeted, record = walk_separable(
        shifted, inputs, [TARGET], step_size=STEP_SIZE, tol=TOL, warn=False
    )
    levels, curves = influence(targeted, inputs, [TARGET])
    estimate = make_estimate(
        ROBUSTNESS_ESTIMAND, float(levels[0]), curves[0], n=len(frame), rule_widened=False
    )
    return _EngineFit(estimate, record)


def robustness_fit(frame: pd.DataFrame, cell: str) -> Any:
    """The fit of one robustness cell."""
    spec = ROBUSTNESS_CELLS[cell]
    if spec is None:
        return study.fit(
            frame, study.CROSS_FITTED, treatment_learner=DummyClassifier(strategy="prior")
        )
    shift_d, shift_y, _ = spec
    return engine_fit(frame, shift_d, shift_y)


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
    return law.sample(ROBUSTNESS_N if family == ROBUSTNESS else CALIBRATION_N, seed)


#: The per-row convergence columns, and the reported summary column they give.
CONVERGENCE_COLUMNS = ("converged", "relative_score", "failure", "phase")
CONVERGENCE_SUMMARY = ("nonconverged_fits",)


def _converged(result: Any) -> dict[str, Any]:
    converged, failure, score, phase = study.convergence(result)
    return {
        "converged": int(converged),
        "relative_score": score,
        "failure": failure,
        "phase": phase,
    }


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, requested = key
    alpha = STUDY.margins.alpha
    if family == ROBUSTNESS:
        fitted = robustness_fit(frame, label)
        return [
            replicate_row(
                property_name=ROBUSTNESS,
                cell=label,
                role=ROBUSTNESS_ROLES[label],
                replicate=replicate,
                n=ROBUSTNESS_N,
                requested=requested,
                truth=law.TRUTH[ROBUSTNESS_ESTIMAND],
                estimate=fitted[ROBUSTNESS_ESTIMAND],
                alpha=alpha,
            )
            | _converged(fitted)
        ]
    scenario, name = CALIBRATION_CELLS[label]
    fitted = study.fit(frame, scenario)
    return [
        replicate_row(
            property_name=CALIBRATION,
            cell=f"{label}__correctly_specified",
            role="positive",
            replicate=replicate,
            n=CALIBRATION_N,
            requested=requested,
            truth=CALIBRATION_TRUTH[label],
            estimate=fitted[name],
            alpha=alpha,
        )
        | _converged(fitted)
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
        (ROBUSTNESS, cell, ROBUSTNESS_ESTIMAND, law.TRUTH[ROBUSTNESS_ESTIMAND])
        for cell in ROBUSTNESS_CELLS
    )
    calibrated = tuple(
        (CALIBRATION, f"{label}__{kind}", name, CALIBRATION_TRUTH[label])
        for label, (_, name) in CALIBRATION_CELLS.items()
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
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *CONVERGENCE_COLUMNS]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts: the union-model rule and the calibration rule with controls."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    return convergence_reading(finish(summary, rates), rows)


#: Whether each fitted property cell's fits are cross-fitted; the derived calibration controls
#: reuse their base cell's fits and are not counted again.
CROSS_FITTED_CELLS: dict[str, bool] = {
    **{cell: spec is None for cell, spec in ROBUSTNESS_CELLS.items()},
    **{
        f"{label}__correctly_specified": scenario == study.CROSS_FITTED
        for label, (scenario, _) in CALIBRATION_CELLS.items()
    },
}


def convergence_suspects(rows: pd.DataFrame) -> list[str]:
    """:func:`~tests.studies.separable_effects.convergence_suspects` on the fitted cells."""
    fits = rows.loc[rows["cell"].isin(list(CROSS_FITTED_CELLS))]
    return study.convergence_suspects(fits, CROSS_FITTED_CELLS)


def convergence_reading(summary: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Report each cell's unconverged fits; the column decides no verdict.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary.
    rows : pandas.DataFrame
        The replication rows.

    Returns
    -------
    pandas.DataFrame
        ``summary`` with :data:`CONVERGENCE_SUMMARY` filled where the rows carry it.
    """
    if "converged" not in rows.columns:
        return summary
    out = summary.copy()
    out["nonconverged_fits"] = np.nan
    for index in out.index:
        group = rows.loc[
            (rows["property"] == out.loc[index, "property"])
            & (rows["cell"] == out.loc[index, "cell"])
        ]
        kept = group.loc[group["converged"].notna()]
        if not kept.empty:
            out.loc[index, "nonconverged_fits"] = float((kept["converged"] == 0).sum())
    return out


#: The failure accounting's declared rules.  The union-model family reads the shared rule.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),)
