"""Registered evidence for the simultaneous band's per-estimate Student t reference (Fix-2).

``band_reference="t"`` keeps the multiplier critical value ``c`` and gives estimate ``j`` its
own ``c_j = F^{-1}_{t, nu_j}(Phi(c))`` with ``nu_j = max(Kish_j, 5)``, the Kish count of the
centred influence curve.  This study measures that band's joint coverage on fresh seeds, cell
by cell, against the default band on the same fits.  It is the evidence that licenses the
option ``("mammen", "logit", "t")`` on the rows of ``_support_rows.T_BAND_CELLS``.

Declared before the pilot (Fix-2 plan rev 3.1, sections 2 and 5):

* the rule (2.1) and the floor 5 (2.4);
* every cell of :data:`CELLS`, the held-out ``strata_n1000__t_band`` included, with its group,
  its sample size and the rows it would license;
* the band-family rule (2.5): a replicate whose band family holds a constant arm cell is
  refused and counted not covered, whatever band the fit shows.  The property module applies
  it with its own copy of Fix-3's check (:func:`tests.studies.band_t_reference_properties.
  constant_arm_cells`), so the verdicts hold for the code before Fix-3's PR 2;
* the paired controls of 5.2: the normal-reference control of each cell and the pointwise
  joint control.  The normal-reference control's clause is a formula of pilot quantities
  (:func:`magnitude_threshold`, :func:`carries_witness`), so the pilot's role is mechanical;
* the red-cell reading (5.5): a passing cell with every clause of its controls licenses the
  triple on its rows; a cell whose coverage passes but whose clause fails licenses nothing; a
  red positive cell keeps the option experimental on its rows.  Neither outcome changes the
  default band.  A failing witness or magnitude clause is a design or implementation failure
  and is diagnosed before any reading.  A high-Kish cell above 0.98 is a finding against the
  rule;
* the held-out cell is a reporting row (``red_cells.REPORTING_ROWS``): judged by the same rule,
  outside the sizing family, licensing no row;
* the replication count: :func:`declared_replicates` on the pilot, the larger of the arm-pooled
  predictive rule over the eleven licensing cells and over the three red shapes, each with
  its clauses (5.3).  :data:`PLANNED_REPLICATES` is that rule on the planning values.

The primary scenario is the shipped ``TMLE()`` default fit of ``default-simultaneous-bands``
with ``band_reference="t"``: the band changes no estimate, SE or pointwise interval, so its
primary rows are that fit's pointwise results against the exact truth.

Nothing here has run.  The study is not in :func:`~tests.studies.evidence.registry.registered`
until its run commits artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_tmle
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.evidence.simultaneous import (
    FAMILY,
    FLOOR_ARMS,
    REFERENCE_CONTROL,
    SCALE_GAIN_FLOOR,
    T_ARMS,
    T_BAND,
    TArm,
    t_cells,
)
from tests.studies.point_study_helpers import primary_rows

SLUG = "band-t-reference"
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 1_000
SEED = 20263621
RESAMPLING_SEED = 20263622
SCENARIO = "t_reference_fit"
ESTIMANDS = ("ate", "att", "atc", "ey1", "ey0", "rr", "or")

#: The pilot's replicates per positive cell and its salted stream (plan 5.1).  Pilot draws
#: never enter a verdict.
PILOT_REPLICATES = 800
PILOT_STREAM = "band-t-reference/pilot"
#: The committed pilot readings, once the pilot has run.
PILOT_PATH = ROOT / "tests" / "studies" / "band_t_reference_pilot.csv"
#: The declared replication count on the planning values (5.3).  The run uses
#: :func:`declared_replicates` on the committed pilot.
PLANNED_REPLICATES = 5_200
#: Above this no R sizes the family; the shortfall goes to the owner before any main run.
MAXIMUM_REPLICATES = 8_000
#: The CPU ceiling of the brief (5.4); a run that would pass it stops and reports.
CPU_CEILING_HOURS = 30.0
#: The smoke run's cells and replicates (two cheap cells at small R).
SMOKE_LABELS = ("strata_n1000", "ordinary_binary")
SMOKE_REPLICATES = 12

#: The normal quantiles of the clause formulas: one-sided 99% and two-sided 99%.
Z99 = float(stats.norm.ppf(0.99))
Z995 = float(stats.norm.ppf(0.995))
#: The magnitude threshold's ceiling (5.2).
MAGNITUDE_CEILING = 0.003

#: The cell groups of 5.2.  ``red``: the magnitude clause.  ``witness``: the nonzero witness
#: where :func:`carries_witness` selects it.  ``no_clause``: no gain bound.  ``held_out``: the
#: witness where selected, outside the sizing family, licensing nothing.
RED = "red"
WITNESS = "witness"
NO_CLAUSE = "no_clause"
HELD_OUT = "held_out"


@dataclass(frozen=True)
class Cell:
    """One ``t`` band cell.

    Parameters
    ----------
    label : str
        The cell prefix.
    source : str
        The fit shape: a ``default_band_properties.SOURCES`` key, ``strata``,
        ``crossfit_strata`` or ``six_regimens``.
    n : int
        The sample size.
    group : str
        :data:`RED`, :data:`WITNESS`, :data:`NO_CLAUSE` or :data:`HELD_OUT`.
    rows : tuple of str
        The support rows the cell would license (``T_BAND_CELLS``).
    planning_coverage : float or None
        The planning coverage of 5.3 (post hoc on the red shapes, guessed elsewhere).
    planning_gain : float or None
        The planning paired gain of 5.3.
    arms : tuple of TArm
        The diagnostic df arms the cell reads.
    """

    label: str
    source: str
    n: int
    group: str
    rows: tuple[str, ...]
    planning_coverage: float | None = None
    planning_gain: float | None = None
    arms: tuple[TArm, ...] = field(default=T_ARMS)


CELLS: tuple[Cell, ...] = (
    Cell("categorical_ltmle", "categorical_ltmle", 2_000, RED, ("G08.in_sample",), 0.9421, 0.009),
    Cell("cde_z0", "cde_z0", 2_000, RED, ("T30",), 0.9458, 0.019),
    Cell("strata", "strata", 2_000, RED, ("T13",), 0.9437, 0.010),
    Cell(
        "strata_n1000",
        "strata",
        1_000,
        HELD_OUT,
        (),
        0.9345,
        None,
        arms=(*T_ARMS, *FLOOR_ARMS),
    ),
    Cell(
        "categorical_ltmle_n8000",
        "categorical_ltmle",
        8_000,
        WITNESS,
        ("G08.in_sample",),
        0.948,
        0.0023,
    ),
    Cell("cde_z0_n8000", "cde_z0", 8_000, WITNESS, ("T30",), 0.949, 0.0048),
    Cell("strata_n8000", "strata", 8_000, WITNESS, ("T13",), 0.950, 0.0025),
    Cell(
        "categorical_ltmle_crossfit",
        "categorical_ltmle_crossfit",
        2_000,
        WITNESS,
        ("G08.cross_fitted",),
        0.951,
        0.009,
    ),
    Cell("crossfit_strata", "crossfit_strata", 2_000, WITNESS, ("T14",), 0.952, 0.010),
    Cell("six_regimens", "six_regimens", 2_000, WITNESS, ("G23.cross_fitted",), 0.955, 0.005),
    Cell("ordinary_binary", "ordinary_binary", 1_000, NO_CLAUSE, ("T01.binary",), 0.953),
    Cell("point_msm", "point_msm", 2_000, NO_CLAUSE, ("T18",), 0.954),
)

CELL_BY_LABEL: dict[str, Cell] = {cell.label: cell for cell in CELLS}
#: The cells that license a row, and so the sizing family (5.3).
LICENSING = tuple(cell for cell in CELLS if cell.group != HELD_OUT)
RED_CELLS = tuple(cell for cell in CELLS if cell.group == RED)


def cell_names(cell: Cell) -> tuple[str, ...]:
    """The declared cells of one label: the ``t`` band, its two controls and its arms."""
    return (*t_cells(cell.label), *(f"{cell.label}__{arm.suffix}" for arm in cell.arms))


PROPERTY_CELLS = {FAMILY: tuple(name for cell in CELLS for name in cell_names(cell))}


# ------------------------------------------------------------------ the clause formulas


def magnitude_threshold(gain: float, sd: float, replicates: int) -> float:
    """``m* = min(0.003, g_p - 2.326 sigma_p - 4.902 sigma_R)`` (plan 5.2).

    ``sigma_p = s_p / sqrt(800)`` and ``sigma_R = s_p / sqrt(R)``: the largest value at most
    0.003 that the clause passes with probability at least 0.99 if the true gain sits at its
    one-sided 99% lower pilot bound.
    """
    sigma_p = sd / math.sqrt(PILOT_REPLICATES)
    sigma_r = sd / math.sqrt(replicates)
    return min(MAGNITUDE_CEILING, gain - Z99 * sigma_p - (Z995 + Z99) * sigma_r)


def carries_witness(gain: float, sd: float, replicates: int) -> bool:
    """Whether a second-group cell carries the nonzero witness (plan 5.2).

    ``g_p - 2.326 sigma_p - 2.576 sigma_R >= 2.326 sigma_R + SCALE_GAIN_FLOOR``.
    """
    sigma_p = sd / math.sqrt(PILOT_REPLICATES)
    sigma_r = sd / math.sqrt(replicates)
    return gain - Z99 * sigma_p - Z995 * sigma_r >= Z99 * sigma_r + SCALE_GAIN_FLOOR


def clause_threshold(cell: Cell, gain: float, sd: float, replicates: int) -> float | None:
    """The lower limit the cell's paired gain must reach, or ``None`` for no clause.

    A red cell whose ``m*`` falls below the witness floor carries the witness only, and the
    design review is told (:func:`witness_only_red_cells`).
    """
    if cell.group == RED:
        return max(magnitude_threshold(gain, sd, replicates), SCALE_GAIN_FLOOR)
    if cell.group in {WITNESS, HELD_OUT}:
        return SCALE_GAIN_FLOOR if carries_witness(gain, sd, replicates) else None
    return None


@dataclass(frozen=True)
class PilotReading:
    """One cell's pilot reading.

    Parameters
    ----------
    coverage : float
        The ``t`` band's gated coverage.
    gain : float
        The mean paired gain over the normal band.
    gain_sd : float
        The per-replicate SD of the paired gain.
    replicates : int
        The pilot replicates the cell kept.
    """

    coverage: float
    gain: float
    gain_sd: float
    replicates: int


def read_pilot(path: Path = PILOT_PATH) -> dict[str, PilotReading] | None:
    """The committed pilot readings by label, or ``None`` before the pilot has run."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return {
        str(row.label): PilotReading(
            float(row.coverage), float(row.gain), float(row.gain_sd), int(row.replicates)
        )
        for row in frame.itertuples()
    }


def reference_thresholds(
    pilot: Mapping[str, PilotReading] | None, replicates: int
) -> dict[str, float | None]:
    """Each label's clause threshold at ``replicates``; empty before the pilot."""
    if pilot is None:
        return {}
    return {
        cell.label: clause_threshold(
            cell, pilot[cell.label].gain, pilot[cell.label].gain_sd, replicates
        )
        for cell in CELLS
    }


def witness_only_red_cells(pilot: Mapping[str, PilotReading], replicates: int) -> list[str]:
    """The red cells whose ``m*`` fell below the witness floor, which the review is told of."""
    return [
        cell.label
        for cell in RED_CELLS
        if magnitude_threshold(pilot[cell.label].gain, pilot[cell.label].gain_sd, replicates)
        < SCALE_GAIN_FLOOR
    ]


# ------------------------------------------------------------------------- the sizing rule
# The arm-pooled predictive rule (guide/studies.md, "Sizing"), as rows/Fix-2/diag/sizing.py
# computes it: normal-normal pooling of the planning coverages, tau at its one-sided 80%
# Q-profile upper bound, the study's Monte Carlo noise drawn, family-wise pass of the CP rule,
# the smallest multiple of 400 with pass probability at least 0.95.  Each gated clause is drawn
# with its gain's planning uncertainty, independently of the coverage draws (an assumption the
# design test states).

SIZING_DRAWS = 40_000
SIZING_SEED = 20261112
SIZING_TARGET = 0.95
SIZING_STEP = 400


def _tau_bound(y: np.ndarray, v: np.ndarray, level: float = 0.80) -> tuple[float, float, float]:
    def q(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = float(np.sum(w * y) / np.sum(w))
        return float(np.sum(w * (y - mu) ** 2))

    crit = float(stats.chi2.ppf(1.0 - level, len(y) - 1))
    t2 = 0.0 if q(0.0) <= crit else float(brentq(lambda t: q(t) - crit, 0.0, 1.0))
    w = 1.0 / (v + t2)
    return math.sqrt(t2), float(np.sum(w * y) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))


def _cp_ok(k: np.ndarray, replicates: int, low: float = 0.92, high: float = 0.98) -> np.ndarray:
    lower = np.where(k > 0, stats.beta.ppf(0.005, k, replicates - k + 1), 0.0)
    upper = np.where(k < replicates, stats.beta.ppf(0.995, k + 1, replicates - k), 1.0)
    return np.asarray((lower >= low) & (upper <= high))


#: One gated clause for the sizing: ``(gain, sd, pilot replicates, cell)``.
Clause = tuple[float, float, int, Cell]


def _clause_pass(
    clauses: Sequence[Clause], replicates: int, draws: int, rng: np.random.Generator
) -> np.ndarray:
    ok = np.ones(draws, dtype=bool)
    for gain, sd, pilot_replicates, cell in clauses:
        threshold = clause_threshold(cell, gain, sd, replicates)
        if threshold is None:
            continue
        true = np.clip(
            gain + sd / math.sqrt(pilot_replicates) * rng.standard_normal(draws), 1e-6, 1
        )
        observed = true + np.sqrt(true * (1 - true) / replicates) * rng.standard_normal(draws)
        clipped = np.clip(observed, 1e-6, 1)
        lower = observed - Z995 * np.sqrt(clipped * (1 - clipped) / replicates)
        ok &= lower >= threshold
    return ok


def family_pass(
    coverages: Sequence[float],
    pilot_replicates: Sequence[int],
    replicates: int,
    *,
    clauses: Sequence[Clause] = (),
    pooled: bool = True,
    draws: int = SIZING_DRAWS,
    seed: int = SIZING_SEED,
) -> float:
    """The family-wise probability that every cell and clause passes at ``replicates``."""
    rng = np.random.default_rng(seed)
    y = np.asarray(coverages, dtype=float)
    v = y * (1 - y) / np.asarray(pilot_replicates, dtype=float)
    if pooled and len(y) > 1:
        tau, mu, s = _tau_bound(y, v)
        m = mu + s * rng.standard_normal(draws)
        shrink = tau**2 / (v + tau**2)
        plan = (
            m[:, None]
            + shrink * (y - m[:, None])
            + np.sqrt(v * shrink) * rng.standard_normal((draws, len(y)))
        )
    else:
        plan = y + np.sqrt(v) * rng.standard_normal((draws, len(y)))
    k = rng.binomial(replicates, np.clip(plan, 0, 1))
    return float(
        np.mean(_cp_ok(k, replicates).all(axis=1) & _clause_pass(clauses, replicates, draws, rng))
    )


def size(
    coverages: Sequence[float],
    pilot_replicates: Sequence[int],
    *,
    clauses: Sequence[Clause] = (),
    ceiling: int = MAXIMUM_REPLICATES,
) -> tuple[int | None, float]:
    """The smallest multiple of 400 up to ``ceiling`` whose family pass reaches 0.95."""
    for replicates in range(SIZING_STEP, ceiling + 1, SIZING_STEP):
        probability = family_pass(coverages, pilot_replicates, replicates, clauses=clauses)
        if probability >= SIZING_TARGET:
            return replicates, probability
    return None, family_pass(coverages, pilot_replicates, ceiling, clauses=clauses)


def _sizing_inputs(
    cells: Sequence[Cell], pilot: Mapping[str, PilotReading] | None
) -> tuple[list[float], list[int], list[Clause]]:
    """Coverages, pilot sizes and clauses: the pilot's, or the planning values of 5.3."""
    coverages: list[float] = []
    sizes: list[int] = []
    clauses: list[Clause] = []
    for cell in cells:
        if pilot is None:
            coverage, size_, gain = cell.planning_coverage, PILOT_REPLICATES, cell.planning_gain
            assert coverage is not None
            sd = None if gain is None else math.sqrt(gain * (1.0 - gain))
        else:
            reading = pilot[cell.label]
            coverage, size_, gain, sd = (
                reading.coverage,
                reading.replicates,
                reading.gain,
                reading.gain_sd,
            )
        coverages.append(float(coverage))
        sizes.append(int(size_))
        if cell.group in {RED, WITNESS} and gain is not None and sd is not None:
            clauses.append((float(gain), float(sd), int(size_), cell))
    return coverages, sizes, clauses


def declared_replicates(pilot: Mapping[str, PilotReading] | None = None) -> tuple[int, float]:
    """The declared R and its family pass: the larger of the two pooled rules of 5.3.

    The eleven licensing cells pooled with every clause the formulas assign, and the three red
    shapes pooled on their own with their magnitude clauses.  ``pilot=None`` evaluates the
    planning values.

    Raises
    ------
    RuntimeError
        If no R up to :data:`MAXIMUM_REPLICATES` sizes either family.
    """
    outcomes = []
    for cells in (LICENSING, RED_CELLS):
        coverages, sizes, clauses = _sizing_inputs(cells, pilot)
        replicates, probability = size(coverages, sizes, clauses=clauses)
        if replicates is None:
            raise RuntimeError(
                f"no R up to {MAXIMUM_REPLICATES} sizes the family (pass {probability:.3f}); "
                "the shortfall goes to the owner before any main run"
            )
        outcomes.append((replicates, probability))
    return max(outcomes)


def held_out_pass(replicates: int, coverage: float, pilot_replicates: int) -> float:
    """The held-out cell's unpooled pass probability at ``replicates``, stated beside it."""
    return family_pass([coverage], [pilot_replicates], replicates, pooled=False)


#: The replication count every cell declares: the pilot's rule once committed, the planning
#: value before.
REPLICATES = PLANNED_REPLICATES if read_pilot() is None else declared_replicates(read_pilot())[0]


STUDY = StudyRecord(
    name="per-estimate t reference for the simultaneous band",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "band_t_reference",
    document="docs/technical-reference/method-evidence/band-t-reference.md",
    anchor="band-t-reference",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-band-t-reference",
    reference=None,
    failure_bound={"cleverly-band-t-reference": 0.01},
    modules=(
        "tests/conftest.py",
        "tests/discrete_law_cde.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/band_t_reference.py",
        "tests/studies/band_t_reference_properties.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/canonical_categorical_ltmle_crossfit.py",
        "tests/studies/canonical_cde_tmle.py",
        "tests/studies/canonical_point_msm.py",
        "tests/studies/canonical_stratified_tmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/studies/default_band_properties.py",
        "tests/studies/default_bands.py",
        "tests/studies/longitudinal_sdr_properties.py",
        "tests/studies/ltmle_crossfit_properties.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/stratified_law.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.band_t_reference",
    properties_module="tests.studies.band_t_reference_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
    property_artifacts=("t-band-replicates.csv",),
)

CONFIGURATION = {
    "primary_fit": (
        "TMLE() with outcome_learner and treatment_learner LogisticRegression(C=1e6, "
        "max_iter=2000) on W1..W3, random_state=0, band_reference='t', every other argument "
        "at its shipped default"
    ),
    "primary_law": "the canonical-tmle binary law (binary_outcome_dgp)",
    "band_rule": "c_j = t.isf(norm.sf(c), max(Kish_j, 5)), Kish of the centred curve",
    "band_cells": (
        "each cell refits its source's subject fit with the default band (Mammen, 10,000 "
        "draws, logit) and reads the t band from the same estimates, draws and c"
    ),
    "cells": {
        cell.label: {"source": cell.source, "n": cell.n, "group": cell.group, "rows": cell.rows}
        for cell in CELLS
    },
    "pilot": f"{PILOT_REPLICATES} replicates per positive cell on stream {PILOT_STREAM!r}",
}

#: The reference control cell of each label, for readers of the design.
REFERENCE_CELLS = {cell.label: f"{cell.label}__{REFERENCE_CONTROL}" for cell in CELLS}
BAND_CELLS = {cell.label: f"{cell.label}__{T_BAND}" for cell in CELLS}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample of the canonical binary law from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return canonical_tmle.draw_from_seed("binary", n, seed)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    """The shipped default fit with the ``t`` band: the learners, seed and reference set."""
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(
            outcome_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            treatment_learner=LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"),
            random_state=0,
            band_reference="t",
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one replication."""
    result = fit_cleverly(frame)
    if tuple(result.estimates) != ESTIMANDS:
        raise AssertionError(f"the primary fit reported {tuple(result.estimates)}")
    if result.simultaneous is None or result.simultaneous.reference != "t":
        raise AssertionError("the primary fit did not build a t band")
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=SCENARIO,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    _, replicate, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(SCENARIO, index, n) for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])
