r"""Repeated-sampling properties of the risk under eliminated competing events.

Two families, fixed before any run.  "Wrong" is ``DummyClassifier(strategy="prior")`` (and a
``DummyRegressor`` for the pseudo outcome), which reads no covariate.

**Double robustness.**  One cell per claim; the design test reads each cell's exact-law limit.

=================================================  ========  ====  =========================
cell                                               role      n     what it witnesses
=================================================  ========  ====  =========================
``tv_rule_t2__mechanism_correct``                  positive  16k   the targeting step: outcome
                                                                   wrong, every factor right.
                                                                   The untargeted plug-in sits
                                                                   1.9 efficient SEs off
``tv_never_t2__outcome_wrong_competing_wrong``     control   16k   the death factor in the
                                                                   clever covariate: its limit
                                                                   sits -0.0199 (1.38 SEs) off
``hb_a1_t2__mechanism_correct``                    positive  16k   the hazard targeting step;
                                                                   the initial hazards' plug-in
                                                                   sits 9.4 SEs off
``hb_a2_t1__outcome_wrong_competing_wrong``        control   16k   the death factor in the
                                                                   hazard clever covariate:
                                                                   1.6 SEs off
=================================================  ========  ====  =========================

``tv`` is the time-varying law with ``continue_if_l2`` the rule plan, ``hb`` the held-baseline
law.  On a saturated law several other configurations coincide with these to rounding (the
outcome-correct fits whatever the death factor, ``SequentialIPW``, ``SDR``); those identities
are unit assertions in ``tests/unit/test_eliminated_competing_events_design.py``, not cells.

**Calibration** (n = 16,000, R = 4,000).  Two positive cells with the two controls derived from
each: ``ltmle_always_t2`` reads ``ate_regimen[always vs never @ t=2]`` on the time-varying law and
``hazard_a1_t3`` reads ``ate_regimen[a1 vs a0 @ t=3]`` on the held-baseline law.  The efficiency
ratios read the exact EIF spread against :data:`EFFICIENCY_RATIO_BAND`.

A replication that raises is never redrawn.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier, DummyRegressor

from tests.parallel import STUDY_JOBS
from tests.studies import eliminated_competing_events as study
from tests.studies import eliminated_competing_law as law
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

ROBUSTNESS_REPLICATES = 1_000
CALIBRATION_N = 16_000
CALIBRATION_REPLICATES = 4_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: ``cell -> (scenario of the fit, n, estimand, configuration, role)``.
ROBUSTNESS_CELLS: dict[str, tuple[str, int, str, str, str]] = {
    "tv_rule_t2__mechanism_correct": (
        study.LTMLE_TV,
        16_000,
        "risk_regimen[continue_if_l2 @ t=2]",
        "mechanism_correct",
        "positive",
    ),
    "tv_never_t2__outcome_wrong_competing_wrong": (
        study.LTMLE_TV,
        16_000,
        "risk_regimen[never @ t=2]",
        "outcome_wrong_competing_wrong",
        "control",
    ),
    "hb_a1_t2__mechanism_correct": (
        study.HAZARD_HB,
        16_000,
        "risk_regimen[a1 @ t=2]",
        "mechanism_correct",
        "positive",
    ),
    "hb_a2_t1__outcome_wrong_competing_wrong": (
        study.HAZARD_HB,
        16_000,
        "risk_regimen[a2 @ t=1]",
        "outcome_wrong_competing_wrong",
        "control",
    ),
}

#: ``label -> (law key, scenario of the fit, estimand)``.
CALIBRATION_CELLS: dict[str, tuple[str, str, str]] = {
    "ltmle_always_t2": ("time_varying", study.LTMLE_TV, "ate_regimen[always vs never @ t=2]"),
    "hazard_a1_t3": ("held_baseline", study.HAZARD_HB, "ate_regimen[a1 vs a0 @ t=3]"),
}
CALIBRATION_LABELS = tuple(CALIBRATION_CELLS)
CALIBRATION_TRUTH = {
    label: study.TRUTH[scenario][name] for label, (_, scenario, name) in CALIBRATION_CELLS.items()
}
EFFICIENCY_SD = {
    label: study.EFFICIENCY_SD[name] for label, (_, _, name) in CALIBRATION_CELLS.items()
}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)


def wrong_learners(configuration: str, scenario: str) -> dict[str, Any]:
    """The learners a robustness configuration replaces."""
    outcome: dict[str, Any] = {"outcome_learner": DummyClassifier(strategy="prior")}
    if scenario != study.HAZARD_HB:
        outcome["pseudo_learner"] = DummyRegressor()
    if configuration == "mechanism_correct":
        return outcome
    if configuration == "outcome_wrong_competing_wrong":
        return {**outcome, "competing_learner": DummyClassifier(strategy="prior")}
    raise KeyError(configuration)


def robustness_fit(frame: pd.DataFrame, cell: str) -> Any:
    """The fit of one robustness cell."""
    scenario, _, _, configuration, _ = ROBUSTNESS_CELLS[cell]
    return study.fit(frame, scenario, **wrong_learners(configuration, scenario))


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
        scenario, n, _, _, _ = ROBUSTNESS_CELLS[label]
    else:
        _, scenario, _ = CALIBRATION_CELLS[label]
        n = CALIBRATION_N
    key_law, _, _ = study.SCENARIOS[scenario]
    return law.tv_sample(n, seed) if key_law == "time_varying" else law.hb_sample(n, seed)


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, requested = key
    alpha = STUDY.margins.alpha
    if family == ROBUSTNESS:
        scenario, n, name, _, role = ROBUSTNESS_CELLS[label]
        return [
            replicate_row(
                property_name=ROBUSTNESS,
                cell=label,
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=study.TRUTH[scenario][name],
                estimate=robustness_fit(frame, label)[name],
                alpha=alpha,
            )
        ]
    _, scenario, name = CALIBRATION_CELLS[label]
    return [
        replicate_row(
            property_name=CALIBRATION,
            cell=f"{label}__correctly_specified",
            role="positive",
            replicate=replicate,
            n=CALIBRATION_N,
            requested=requested,
            truth=CALIBRATION_TRUTH[label],
            estimate=study.fit(frame, scenario)[name],
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
        (ROBUSTNESS, cell, name, study.TRUTH[scenario][name])
        for cell, (scenario, _, name, _, _) in ROBUSTNESS_CELLS.items()
    )
    calibrated = tuple(
        (CALIBRATION, f"{label}__{kind}", name, CALIBRATION_TRUTH[label])
        for label, (_, _, name) in CALIBRATION_CELLS.items()
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
