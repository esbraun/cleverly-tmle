r"""Repeated-sampling properties of the kernel-smoothed dose-response curve (X93 S1).

Three families, fixed before any run, each on its own seed stream
(``stream_seed(STUDY, "property_sample", family, label, r)``).  Every cell fits the declared
binary configuration of :mod:`tests.studies.dose_response` at n = 4,000.  The calibration and
robustness cells declare only the pair they read; every nuisance fit is the binary scenario's,
and the pair's estimate differs from the 15-pair fit's only by the quadrature partition, whose
doubling diagnostic the fit reports.

**Simultaneous coverage** (``binary_curve``, R = :data:`BAND_REPLICATES`).  One fit of the
15-pair binary declaration gives two rows through
:func:`tests.studies.evidence.simultaneous.joint_coverage_rows`: the band row, covered when
every truth lies inside the package's multiplier band, and ``pointwise_joint_control``,
covered when every truth lies inside its own pointwise interval.  The band's 99% joint
coverage interval must lie inside [0.92, 0.98]; the control's upper endpoint must fall below
0.95.  The design computation (``tests/unit/test_dose_response_design.py``) reads the
family's exact efficient-influence correlation off the law
(:func:`tests.studies.dose_response_law.efficiency_covariance`), so ``p0`` is not a design
estimate.

**Interval calibration** (``interior_h1_a2_5``, R = :data:`CALIBRATION_REPLICATES`).  The pair
``h = 1.0, a0 = 2.5``, an interior pair at the curve's curvature, with the two derived controls
of :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the reported SE times
:data:`SHRUNKEN_SE_FACTOR`, and one efficiency unit of independent noise added to each estimate.
:data:`EFFICIENCY_SD` is the exact SD of the pair's efficient influence curve
(:func:`~tests.studies.dose_response_law.efficiency_sd`, checked against a 2,000,000-draw Monte
Carlo in the design test).

**Double robustness** (pair ``h = 1.0, a0 = 3.5``, R = :data:`ROBUSTNESS_REPLICATES`).
"Density wrong" is the hazard classifier on the bin indicators alone, so the fitted density is
the binned marginal density of the dose and ignores :math:`W`.

==============================================  ========  ===============================================
cell                                            role      what it witnesses
==============================================  ========  ===============================================
``h1_a3_5__outcome_correct_density_wrong``      positive  the outcome regression carries the fit when the
                                                          density is wrong: the declared sieve, whose
                                                          limit sits 0.017 efficient SEs (n = 4,000) off
                                                          the truth through the spline error in f(a)
``h1_a3_5__both_wrong``                         control   both nuisances ignore the confounder: the
                                                          outcome is the dose-only spline
                                                          (``DoseSieve(interactions=False)``), and the
                                                          limit is the unadjusted kernel regression of Y
                                                          on A, 3.03 efficient SEs off the truth
==============================================  ========  ===============================================

The limits are the design test's population computation (a 1,000,000-row fixed-seed sample
for the regressions, the binned marginal density at the 32 bins of n = 4,000, and the
fluctuation solved on the population).  The control's first design, a main-terms logistic
regression on ``[A, W]``, was rejected before any run: its limit sits below 0.65 efficient SEs
off the truth at every declared pair (0.40 at this one), too close for the discrimination floor
at R = 800.

**Declared blind spot.**  No cell measures "outcome wrong, density right".  With the binned
density consistent but not at a known rate, the remainder is first order in the binned
density's error, so the estimate is consistent but not root-n, and a bias interval at one n
cannot separate a slowly vanishing bias from a defect.  The unit tests of the exact remainder
(``tests/unit/test_dose_response.py``) carry that direction.

**Reported, not gated: the claim boundary.**  The pointwise interval read against the
unsmoothed :math:`\theta(a_0)` at the curvature dose 2.5 is expected to under-cover by
:math:`\Phi(1.96 - b/\mathrm{SE}) - \Phi(-1.96 - b/\mathrm{SE})` with
:math:`b = \theta_h(2.5) - \theta(2.5)` and the efficient SE at n = 4,000: 0.938 at h = 0.5
(the declared bandwidth nearest the exact MSE-optimal 0.60), 0.597 at h = 1.0, and 0.010 at
h = 1.5.  No shared family reads a coverage that must fall below nominal against a second
truth, so the quantity is computed from the binary scenario's primary rows by the design test's
formula and reported, not gated.

**Sizes** (declared before any run, from :mod:`tests.studies.dose_response_pilot`; the design
test recomputes each).  Each sized term carries half the shared
:data:`~tests.studies.repeated_sizing.SIZED_FAIL`.

* :data:`BAND_REPLICATES` = 7,600: the smallest multiple of 400 at which
  :func:`~tests.studies.repeated_sizing.joint_band_pass_probability` reaches that share at the
  one-sided 80% lower bound (0.935) of the pilot's joint band coverage, 942 of 1,000.
* :data:`CALIBRATION_REPLICATES` = 6,000: the smallest multiple of 400 at which
  :func:`~tests.studies.repeated_sizing.calibration_pass` reaches that share at the pooled pilot
  SE ratio 1.023 of the calibration pair (the single-pair pilot reads 1.052, the same pair in
  the 15-pair pilot 0.996).  Read with that ratio's pilot noise the pass probability is 0.89,
  not the share: the reported SE is conservative by a few percent at n = 4,000, and the pilot
  cannot exclude a ratio near the band's upper end, 1.07.  ``regenerate.py`` declares the route
  for a red SE ratio with green coverage and efficiency ratios.
* :data:`ROBUSTNESS_REPLICATES` = 800: the positive's pass probability at the one-sided 80%
  upper bound (0.059) of its pilot standardized bias first exceeds
  :data:`~tests.studies.repeated_sizing.SIZED_SHARE` at 800 on a grid of 200; the control's
  pilot bias is 2.87 SDs, discriminated with probability above the floor at any of these sizes.

A replication that raises is never redrawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.stats import norm

from cleverly.dose_response import DoseResponse, pair_name
from tests.parallel import STUDY_JOBS
from tests.studies import dose_response as study
from tests.studies import dose_response_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as SIMULTANEOUS
from tests.studies.evidence.simultaneous import joint_cells, joint_coverage_rows

STUDY = study.STUDY

CALIBRATION = "interval_calibration"
ROBUSTNESS = "double_robustness"

PROPERTY_N = study.PRIMARY_N
BAND_REPLICATES = 7_600
CALIBRATION_REPLICATES = 6_000
ROBUSTNESS_REPLICATES = 800
CALIBRATION_N = PROPERTY_N

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

BAND_LABEL = "binary_curve"
CALIBRATION_LABEL = "interior_h1_a2_5"
CALIBRATION_PAIR = (1.0, 2.5)
ROBUSTNESS_PAIR = (1.0, 3.5)
CALIBRATION_LABELS = (CALIBRATION_LABEL,)


def single_pair(h: float, a0: float) -> DoseResponse:
    """The binary declaration restricted to one pair."""
    return DoseResponse((a0,), (h,), study.SUPPORT)


CALIBRATION_NAME = pair_name(*CALIBRATION_PAIR)
ROBUSTNESS_NAME = pair_name(*ROBUSTNESS_PAIR)
CALIBRATION_TRUTH = {CALIBRATION_LABEL: law.smoothed(*CALIBRATION_PAIR)}
CALIBRATION_TRUTH_BY_NAME = {CALIBRATION_NAME: CALIBRATION_TRUTH[CALIBRATION_LABEL]}
ROBUSTNESS_TRUTH = law.smoothed(*ROBUSTNESS_PAIR)
EFFICIENCY_SD = {CALIBRATION_LABEL: law.efficiency_sd(*CALIBRATION_PAIR)}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)


@dataclass(frozen=True)
class RobustnessCell:
    """One double-robustness cell.

    Parameters
    ----------
    role : str
        ``"positive"`` or ``"control"``.
    outcome_correct : bool
        Whether the outcome regression is the declared sieve, else the dose-only spline.
    """

    role: str
    outcome_correct: bool


ROBUSTNESS_CELLS: dict[str, RobustnessCell] = {
    "h1_a3_5__outcome_correct_density_wrong": RobustnessCell("positive", outcome_correct=True),
    "h1_a3_5__both_wrong": RobustnessCell("control", outcome_correct=False),
}


def robustness_learners(cell: str) -> dict[str, Any]:
    """The learners a robustness cell replaces: the marginal density, and its outcome."""
    spec = ROBUSTNESS_CELLS[cell]
    return {
        "outcome_learner": study.outcome_learner(interactions=spec.outcome_correct),
        "treatment_learner": study.hazard_learner(len(study.COVARIATES), form="marginal"),
    }


def robustness_fit(frame: pd.DataFrame, cell: str) -> Any:
    """The fit of one robustness cell."""
    return study.fit(
        frame, study.BINARY, curve=single_pair(*ROBUSTNESS_PAIR), **robustness_learners(cell)
    )


def calibration_fit(frame: pd.DataFrame) -> Any:
    """The fit of the calibration cell: the declared binary configuration at its pair."""
    return study.fit(frame, study.BINARY, curve=single_pair(*CALIBRATION_PAIR))


def band_fit(frame: pd.DataFrame) -> Any:
    """The fit of the band cell: the declared binary scenario."""
    return study.fit(frame, study.BINARY)


@dataclass(frozen=True)
class _Banded:
    """A dose-response result read through the joint-coverage interface."""

    result: Any

    @property
    def simultaneous(self) -> Any:
        return self.result.band

    def __getitem__(self, name: str) -> Any:
        return self.result.estimates[name]


# ---------------------------------------------------------------------------- keys

#: ``(family, label, replicate, requested)``: a band label, a calibration label, or a cell.
_Key = tuple[str, str, int, int]


def _budget(declared: int, budget: int | None) -> int:
    return declared if budget is None else min(budget, declared)


def _keys(budget: int | None) -> list[_Key]:
    band = _budget(BAND_REPLICATES, budget)
    calibrate = _budget(CALIBRATION_REPLICATES, budget)
    robust = _budget(ROBUSTNESS_REPLICATES, budget)
    return [
        *((SIMULTANEOUS, BAND_LABEL, r, band) for r in range(band)),
        *((CALIBRATION, CALIBRATION_LABEL, r, calibrate) for r in range(calibrate)),
        *((ROBUSTNESS, cell, r, robust) for cell in ROBUSTNESS_CELLS for r in range(robust)),
    ]


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, _ = key
    return law.sample(PROPERTY_N, stream_seed(STUDY, "property_sample", family, label, replicate))


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, requested = key
    alpha = STUDY.margins.alpha
    if family == SIMULTANEOUS:
        result = band_fit(frame)
        names = study.ESTIMANDS[study.BINARY]
        return joint_coverage_rows(
            _Banded(result),
            study.TRUTH[study.BINARY],
            names,
            label=label,
            replicate=replicate,
            n=PROPERTY_N,
            requested=requested,
            pointwise_critical=CRITICAL,
        )
    if family == CALIBRATION:
        return [
            replicate_row(
                property_name=CALIBRATION,
                cell=f"{label}__correctly_specified",
                role="positive",
                replicate=replicate,
                n=CALIBRATION_N,
                requested=requested,
                truth=CALIBRATION_TRUTH[label],
                estimate=calibration_fit(frame)[CALIBRATION_NAME],
                alpha=alpha,
            )
        ]
    return [
        replicate_row(
            property_name=ROBUSTNESS,
            cell=label,
            role=ROBUSTNESS_CELLS[label].role,
            replicate=replicate,
            n=PROPERTY_N,
            requested=requested,
            truth=ROBUSTNESS_TRUTH,
            estimate=robustness_fit(frame, label)[ROBUSTNESS_NAME],
            alpha=alpha,
        )
    ]


def _cells_of(key: _Key) -> list[str]:
    family, label, _, _ = key
    if family == SIMULTANEOUS:
        return [f"{SIMULTANEOUS}/{cell}" for cell in joint_cells(label)]
    if family == CALIBRATION:
        name = f"{CALIBRATION}/{label}__correctly_specified"
        return [name, *DERIVED_CELLS[name]]
    return [f"{ROBUSTNESS}/{label}"]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries."""
    joint = tuple((SIMULTANEOUS, cell, "max_t", 0.0) for cell in joint_cells(BAND_LABEL))
    calibrated = tuple(
        (CALIBRATION, f"{CALIBRATION_LABEL}__{kind}", CALIBRATION_NAME, CALIBRATION_TRUTH[label])
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    )
    robust = tuple(
        (ROBUSTNESS, cell, ROBUSTNESS_NAME, ROBUSTNESS_TRUTH) for cell in ROBUSTNESS_CELLS
    )
    return joint + calibrated + robust


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
    """The shared verdicts: the union-model rule, the calibration rule and the band rule."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)


#: The failure accounting's declared rules.  The union-model and band families read the shared
#: rules.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),)
