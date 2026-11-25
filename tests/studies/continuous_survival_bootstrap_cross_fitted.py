r"""Reported evidence for the cross-fitted full-refit bootstrap of the continuous-time TMLE.

The subject is the default, cross-fitted :class:`~cleverly.survival.ContinuousSurvivalTMLE` with
``n_bootstrap=500``, on the base law at :math:`n = 1{,}000`, built exactly as the gated in-sample
record :mod:`tests.studies.continuous_survival_bootstrap` builds its fits (the engine, the draws,
the readings and the calibration table are that module's).  It is a record of its own so that it
runs at the plan's 1,000 replicates rather than the in-sample record's 3,200, and so that no
reading of it can gate anything, now or after a run:

* the one cell, ``b2_cross_fitted``, is a reported scenario of a ``reporting`` record, its failure
  bound is published but gates nothing (``reported_bounds_gate=False``), and
  :func:`calibration` gives none of its rows a verdict;
* no theorem covers a bootstrap of a cross-fitted fit with a selected library, and Cai and van der
  Laan (2020) warn against it, so no reading moves the cross-fitted label.

**Rows.** Every risk and risk difference at the three target times (0.46, 1.12, 2.32), 18 rows,
each with the percentile, BCa and Wald readings of the same fit.

**Size and seeds.** :data:`PRIMARY_REPLICATES` is the plan's 1,000.  :data:`SEED` and
:data:`RESAMPLING_SEED` are used by no other study; replicate ``r`` draws from
``replicate_seed(STUDY, "b2_cross_fitted", r)`` and seeds its bootstrap with the same value.  The
pilot's ``cross_fitted`` draws (:mod:`tests.studies.continuous_survival_bootstrap_pilot`), on
their own streams, give this record's failure and seconds readings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival_bootstrap as gated
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

#: The plan's count for the cross-fitted cell (5.5).
PRIMARY_REPLICATES = 1_000
#: The pilot of the cross-fitted fit, which reads its failures and seconds only.
PILOT_REPLICATES = 50
PRIMARY_N = gated.PRIMARY_N
N_BOOTSTRAP = gated.N_BOOTSTRAP
FAILURE_BOUND = gated.FAILURE_BOUND
SEED = 2026612071
RESAMPLING_SEED = 2026612072

SUBJECT = "cleverly-continuous-survival-bootstrap-cross-fitted"

CELL = "b2_cross_fitted"
#: ``cell -> cross-fitted``.
CELLS: dict[str, bool] = {CELL: True}
#: Every risk and risk difference at the three target times, in the engine's names.
ROWS: tuple[str, ...] = tuple(law.TRUTH)
ESTIMANDS: dict[str, tuple[str, ...]] = {CELL: ROWS}

CALIBRATION_FILE = gated.CALIBRATION_FILE
READINGS_FILE = gated.READINGS_FILE

STUDY = StudyRecord(
    name="continuous-time survival full-refit bootstrap, cross-fitted",
    slug="continuous-survival-bootstrap-cross-fitted",
    artifacts=ROOT / "tests" / "canonical" / "continuous_survival_bootstrap_cross_fitted",
    document="docs/technical-reference/method-evidence/continuous-survival-bootstrap.md",
    anchor="continuous-time-survival-full-refit-bootstrap-cross-fitted",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=gated.MARGINS,
    implementation=SUBJECT,
    reference=None,
    reported_scenarios=frozenset(CELLS),
    reported_bounds_gate=False,
    publication_policy="reporting",
    failure_bound={SUBJECT: FAILURE_BOUND},
    nuisance_count=3,
    extra_artifacts=(CALIBRATION_FILE, READINGS_FILE),
    modules=(
        "tests/studies/continuous_survival_bootstrap_cross_fitted.py",
        *gated.STUDY.modules,
    ),
    runner_module="tests.studies.continuous_survival_bootstrap_cross_fitted",
    properties_module="tests.studies.continuous_survival_bootstrap_properties",
)

CONFIGURATION = {
    **gated.CONFIGURATION,
    "cells": {CELL: "cross-fitted"},
    "interval": "percentile, BCa and Wald on the same fits (all reported)",
    "rows": "risks and differences at the three quartiles (reported)",
    "sizes": {CELL: PRIMARY_N},
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return gated.draw_from_seed(scenario, n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    return gated.fit_replicate(STUDY, CELLS[key[0]], ROWS, False, key, drawn)


#: The per-row readings beside the published schema.
EXTRA_COLUMNS = (*gated.INTERVAL_COLUMNS, *gated.FIT_COLUMNS)


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    scenarios: tuple[str, ...] | None = None,
) -> replication.PhaseResult:
    """The cross-fitted cell at its declared size."""
    return gated.fit_cells(
        STUDY,
        _replicate,
        _draw,
        EXTRA_COLUMNS,
        replicates=replicates,
        n=n,
        n_jobs=n_jobs,
        scenarios=scenarios,
    )


def calibration(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration table of :func:`~tests.studies.continuous_survival_bootstrap.calibration`:
    every row reported, read by the rule with no verdict."""
    return gated.calibration(rows, STUDY)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The calibration readings and the targeting record, neither a verdict."""
    return {
        CALIBRATION_FILE: calibration(rows),
        READINGS_FILE: gated.targeting_readings(rows),
    }


def scientific_failures(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """No reading of this record is gated."""
    del artifacts
    return {}


def account_failures(
    failures: pd.DataFrame | None, artifacts: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """No scientific gate, so a lost fit counts against the published bound only."""
    return evidence_failures.audits({}, missing={})
