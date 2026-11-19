r"""Calibration and double robustness of ``continuous-survival-causes`` (X6 part (b)).

Fixed before any run (X6 plan rev 5, section 5.3).  Every fit is cross-fitted with the study's 10
outer and 5 inner folds at :math:`n = 1{,}000` on the laws of
:mod:`tests.studies.continuous_survival_causes_law`.

**Calibration** (:data:`CALIBRATION_REPLICATES` each, on the stream
``stream_seed(STUDY, "property_sample", family, label, r)``).  One positive cell per law, each
the primary configuration of :mod:`tests.studies.continuous_survival_causes`:
``l1_rd__correctly_specified`` reads L1's risk difference at its median and
``l3_rd__correctly_specified`` the risk difference of L3's third cause, the cause this study adds,
at its median (:data:`CALIBRATION_TARGETS`).  Each reads the shared calibration band (SE ratio's
99% interval inside (0.93, 1.07), coverage's inside (0.92, 0.98)).  The two shared controls are
derived from each cell's rows: ``shrunken_se_control`` multiplies every standard error by
:data:`SHRUNKEN_SE_FACTOR` (0.85, the plan's control) and ``noise_control`` adds one efficient
standard deviation of noise (used only to size it: the cell claims no efficiency).  The count is
the arm-pooled predictive rule on the pilot
(:func:`tests.studies.continuous_survival_causes_pilot.calibration_count`).

**Double robustness** (:data:`ROBUSTNESS_REPLICATES` draws per law, on the stream
``stream_seed(STUDY, "property_sample", "double_robustness", law, r)``).  Each draw is fitted twice
with the wrong event hazards (the Cox model on the treatment alone, which ignores every
confounder) and the right propensity and censoring (the default learners hold both laws'
logistic propensity and constant censoring): once with the shipped tilt and once with
``tilt="factual"``.  Per label (:data:`ROBUSTNESS_LABELS`: the treated arm's cause-1 risk at the
median of each law, and L3's treated cause-3 risk there):

=================================  ========  =================================================
cell                               role      estimate
=================================  ========  =================================================
``<label>__mechanism_correct``     positive  the targeted estimate: the shipped TMLE is doubly
                                             robust, so its bias must lie inside 0.2 SD
``<label>__plug_in_control``       control   the initial plug-in of the same fit
                                             (``initial_estimates``): targeting is necessary,
                                             so its bias must lie beyond 0.2 SD
``<label>__factual_tilt_control``  control   the ``tilt="factual"`` fit of the same draw (the
                                             ``concrete`` 1.0.5 update): not doubly robust,
                                             so its bias must lie beyond 0.2 SD
=================================  ========  =================================================

The plug-in has no standard error of its own; its row carries the targeted fit's, which the
union-model scale screen (``property_verdicts.UNION_MODEL_SE_BAND``, 0.1 to 10) reads and the
bias verdict does not.  A control whose pilot pass probability at 1,000 draws is below 0.80 is
declared uninformative at that law before the run (``regenerate.py``), and then the three-cause
tilt rests on the exact-law tests of ``tests/unit/test_continuous_survival_causes.py``.

A replication that raises is never redrawn.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_causes as study
from tests.studies import continuous_survival_causes_law as laws
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

ROBUSTNESS_N = study.PRIMARY_N
#: The plan's control draws per law (5.3).
ROBUSTNESS_REPLICATES = 1_000
CALIBRATION_N = study.PRIMARY_N
#: The arm-pooled predictive rule's count on the committed pilot (``pilot.calibration_count``).
CALIBRATION_REPLICATES = 3_200
SHRUNKEN_SE_FACTOR = 0.85
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

_L1, _L3 = laws.LAWS["one_cause"], laws.LAWS["three_causes"]

#: ``label -> (primary cell, estimand)`` of each calibration cell.
CALIBRATION_TARGETS: dict[str, tuple[str, str]] = {
    "l1_rd": ("l1_cf_1k", _L1.difference_name(1, _L1.times[1])),
    "l3_rd": ("l3_cf_1k", _L3.difference_name(3, _L3.times[1])),
}
CALIBRATION_LABELS = tuple(CALIBRATION_TARGETS)
EFFICIENCY_SD = {
    "l1_rd": _L1.efficiency_sd(1, _L1.times[1])["difference"],
    "l3_rd": _L3.efficiency_sd(3, _L3.times[1])["difference"],
}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

#: ``law -> {label: estimand}`` of the double-robustness cells.
ROBUSTNESS_LABELS: dict[str, dict[str, str]] = {
    "one_cause": {"l1_c1": _L1.level_name("treated", 1, _L1.times[1])},
    "three_causes": {
        "l3_c1": _L3.level_name("treated", 1, _L3.times[1]),
        "l3_c3": _L3.level_name("treated", 3, _L3.times[1]),
    },
}
#: The double-robustness controls the committed pilot reads as uninformative (pass probability
#: at :data:`ROBUSTNESS_REPLICATES` draws below 0.80): declared before the run, checked by
#: ``regenerate.py`` against the pilot.
UNINFORMATIVE_CONTROLS: tuple[str, ...] = ()
#: The two fits of each double-robustness draw.
TILTS = ("all", "factual")


def _law_of_cell(cell: str) -> str:
    return study.CELLS[cell]


# ---------------------------------------------------------------------------- keys

#: ``(family, label, replicate, requested, tilt)``: a law's robustness fit with one tilt, or a
#: calibration label (tilt ``"all"``).
_Key = tuple[str, str, int, int, str]


def _keys(budget: int | None) -> list[_Key]:
    robust = ROBUSTNESS_REPLICATES if budget is None else min(budget, ROBUSTNESS_REPLICATES)
    calibrate = CALIBRATION_REPLICATES if budget is None else min(budget, CALIBRATION_REPLICATES)
    return [
        *(
            (ROBUSTNESS, name, r, robust, tilt)
            for name in ROBUSTNESS_LABELS
            for tilt in TILTS
            for r in range(robust)
        ),
        *(
            (CALIBRATION, label, r, calibrate, "all")
            for label in CALIBRATION_LABELS
            for r in range(calibrate)
        ),
    ]


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, _, _ = key
    # Both tilts of one robustness draw read the same sample.
    seed = stream_seed(STUDY, "property_sample", family, label, replicate)
    if family == ROBUSTNESS:
        return laws.LAWS[label].sample(ROBUSTNESS_N, seed)
    cell, _ = CALIBRATION_TARGETS[label]
    return study.law_of(cell).sample(CALIBRATION_N, seed)


def _robustness_rows(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, name, replicate, requested, tilt = key
    law = laws.LAWS[name]
    truth = laws.truth(name)
    result = registered.fit_with(frame, study.engine(law, hazard_wrong=True, tilt=tilt))
    rows = []
    for label, estimand in ROBUSTNESS_LABELS[name].items():
        estimate = result[estimand]
        common = {
            "property_name": ROBUSTNESS,
            "replicate": replicate,
            "n": ROBUSTNESS_N,
            "requested": requested,
            "truth": truth[estimand],
            "alpha": STUDY.margins.alpha,
        }
        if tilt == "factual":
            rows.append(
                replicate_row(
                    cell=f"{label}__factual_tilt_control",
                    role="control",
                    estimate=estimate,
                    **common,
                )
            )
            continue
        rows.append(
            replicate_row(
                cell=f"{label}__mechanism_correct", role="positive", estimate=estimate, **common
            )
        )
        plug_in = replicate_row(
            cell=f"{label}__plug_in_control", role="control", estimate=estimate, **common
        )
        # The plug-in's point estimate with the targeted fit's standard error and interval: the
        # bias verdict reads the estimate, the scale screen the standard error.
        value = float(result.initial_estimates[estimand])
        plug_in["estimate"] = value
        half = CRITICAL * plug_in["std_error"]
        plug_in["covered"] = int(value - half <= truth[estimand] <= value + half)
        plug_in["rejected"] = int(abs(value) > half)
        rows.append(plug_in)
    return rows


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, requested, _ = key
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == ROBUSTNESS:
            return _robustness_rows(key, frame)
        cell, estimand = CALIBRATION_TARGETS[label]
        estimate = study.fit(frame, cell)[estimand]
    return [
        replicate_row(
            property_name=CALIBRATION,
            cell=f"{label}__correctly_specified",
            role="positive",
            replicate=replicate,
            n=CALIBRATION_N,
            requested=requested,
            truth=laws.truth(_law_of_cell(cell))[estimand],
            estimate=estimate,
            alpha=STUDY.margins.alpha,
        )
    ]


def _cells_of(key: _Key) -> list[str]:
    family, label, _, _, tilt = key
    if family == ROBUSTNESS:
        kinds = (
            ("factual_tilt_control",)
            if tilt == "factual"
            else (
                "mechanism_correct",
                "plug_in_control",
            )
        )
        return [
            f"{ROBUSTNESS}/{name}__{kind}" for name in ROBUSTNESS_LABELS[label] for kind in kinds
        ]
    name = f"{CALIBRATION}/{label}__correctly_specified"
    return [name, *DERIVED_CELLS[name]]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries."""
    robust = tuple(
        (ROBUSTNESS, f"{label}__{kind}", estimand, laws.truth(name)[estimand])
        for name, labels in ROBUSTNESS_LABELS.items()
        for label, estimand in labels.items()
        for kind in ("mechanism_correct", "plug_in_control", "factual_tilt_control")
    )
    calibrated = tuple(
        (CALIBRATION, f"{label}__{kind}", estimand, laws.truth(_law_of_cell(cell))[estimand])
        for label, (cell, estimand) in CALIBRATION_TARGETS.items()
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
    """The shared verdicts: the union-model rule and the calibration rule with its controls."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The union-model family reads the shared rule.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)


def efficient_scale(label: str) -> float:
    """The efficient standard deviation of a calibration label's estimate over root n."""
    return EFFICIENCY_SD[label] / float(np.sqrt(CALIBRATION_N))
