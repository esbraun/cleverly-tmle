r"""Repeated-sampling properties of the kernel-smoothed dose-response curve (X93 S1).

Three families, fixed before any run.  Every cell fits the binary law of
:mod:`tests.studies.dose_response` at n = 4,000.  The band and calibration families read one
set of fits on the stream ``stream_seed(STUDY, "property_sample", "simultaneous_coverage",
"binary_curve", r)``, as the X32b calibration instrument reads its C4 band off its calibration
fits.  The robustness cells read one shared stream, ``stream_seed(STUDY, "property_sample",
"double_robustness", "h1_a3_5", r)``, so every configuration is fitted on the same draws.

**Simultaneous coverage** (``binary_curve``, R = :data:`CURVE_REPLICATES`).  One fit of the
15-pair binary declaration gives two rows through
:func:`tests.studies.evidence.simultaneous.joint_coverage_rows`: the band row, covered when
every truth lies inside the package's multiplier band, and ``pointwise_joint_control``,
covered when every truth lies inside its own pointwise interval.  The band's 99% joint
coverage interval must lie inside [0.92, 0.98]; the control's upper endpoint must fall below
0.95.  The design computation (``tests/unit/test_dose_response_design.py``) reads the
family's exact efficient-influence correlation off the law
(:func:`tests.studies.dose_response_law.efficiency_covariance`), so ``p0`` is not a design
estimate.  The fit seeds the multiplier draws with ``random_state=0``, so every replicate reads
the same multiplier draws, and the cell measures the band's coverage conditional on that one
draw: since band-finite-sample, 10,000 Mammen draws on the identity scale (the curve's estimates
carry no range, so the logit scale maps nothing), and before it 1,000 draws.  At 1,000 draws the
critical value was the oracle's to about 0.02 (the pilot's mean is 2.733 against the exact
2.751), which moves the joint coverage by about 0.003.

**Interval calibration** (``interior_h1_a2_5``, R = :data:`CURVE_REPLICATES`).  The pair
``h = 1.0, a0 = 2.5`` of each band fit, an interior pair at the curve's curvature, with the two
derived controls of :func:`~tests.studies.evidence.property_verdicts.calibration_controls`: the
reported SE times :data:`SHRUNKEN_SE_FACTOR`, and one efficiency unit of independent noise added
to each estimate.  :data:`EFFICIENCY_SD` is the exact SD of the pair's efficient influence curve
(:func:`~tests.studies.dose_response_law.efficiency_sd`, checked against a 2,000,000-draw Monte
Carlo in the design test).  A fit of the pair alone shares every nuisance fit and the
fluctuation with the 15-pair fit and differs only in the quadrature partition: on the 20 shared
draws of the pilot's ``agreement_4k`` label the two estimates agree to below 1e-6 of the
reported SE, so the single-pair pilot fits and the band fits measure one estimator.

**Double robustness** (pair ``h = 1.0, a0 = 3.5``, R = :data:`ROBUSTNESS_REPLICATES`).
"Density wrong" is the hazard classifier on the bin indicators alone, so the fitted density is
the binned marginal density of the dose and ignores :math:`W`.  "Outcome wrong" is the
dose-only spline (``DoseSieve(interactions=False)``), which ignores :math:`W`.

==================================================  ==========  ==================================
cell                                                role        what it witnesses
==================================================  ==========  ==================================
``h1_a3_5__outcome_correct_density_wrong``          positive    the outcome regression carries the
                                                                fit when the density is wrong: the
                                                                declared sieve, whose limit sits
                                                                0.017 efficient SEs (n = 4,000)
                                                                off the truth through the spline
                                                                error in f(a)
``h1_a3_5__both_wrong``                             control     both nuisances ignore the
                                                                confounder; the limit is the
                                                                unadjusted kernel regression of Y
                                                                on A, 3.03 efficient SEs off
``h1_a3_5__outcome_wrong_density_declared``         diagnostic  the targeted estimate with the
                                                                dose-only outcome and the declared
                                                                density
``h1_a3_5__outcome_wrong_density_declared_plug_in`` diagnostic  the untargeted plug-in of that fit
==================================================  ==========  ==================================

The positive cannot detect a targeting defect: with the outcome right the fluctuation is near
zero, so an untargeted plug-in, a dropped :math:`1/g` or :math:`g` read at :math:`a_0` would each
pass it.  The targeting direction is the third configuration.  Its two rows publish their
biases, and the family's joint clause gates the **targeting gain** on them: with
:math:`b_\text{plug-in}` and :math:`b_\text{targeted}` the mean errors of the two rows in units of
the exact efficient SE at n = 4,000 (:data:`ROBUSTNESS_SE`), both
:math:`b_\text{plug-in} - b_\text{targeted}` (``targeting_gain``) and
:math:`b_\text{plug-in} + b_\text{targeted}` (``targeting_gain_past_truth``) must be at least
:data:`TARGETING_GAIN_FLOOR`.  Together they say
:math:`|b_\text{targeted}| \le b_\text{plug-in} - 1.5`: the targeted estimate sits at least 1.5
SEs closer to the truth than its plug-in, on either side of it.  An untargeted fit reads 0 on the
first, and the pilot (``dr_targeting_4k``) reads 3.32 and 2.49.  The joint clause is ANDed into
the positive's and the control's ``property_passed``, so a red gain turns the family red.

**Why no equivalence positive for "outcome wrong".**  The declared hazard classifier is a fixed
parametric logistic model, ``[indicators, W, W^2, bin * W, bin * W^2]``.  The true hazard of a
normal location model with a tanh mean is not in that model, so the fitted density converges to
a projection, not to :math:`g_0`, and the remainder of the outcome-wrong fit does not vanish.
The pilot's targeted bias, -0.42 efficient SEs against a plug-in bias of 2.90, is that
misspecification plus binning.  A bias interval around the truth would therefore fail for a
correct estimator, and the cell gates the gain instead.  The unit tests of the exact remainder
(``tests/unit/test_dose_response.py``) carry the population statement with the true density.

The limits are the design test's population computation (a 1,000,000-row fixed-seed sample
for the regressions, the binned marginal density at the 32 bins of n = 4,000, and the
fluctuation solved on the population).  The control's first design, a main-terms logistic
regression on ``[A, W]``, was rejected before any run: its limit sits below 0.65 efficient SEs
off the truth at every declared pair (0.40 at this one), too close for the discrimination floor
at R = 800.

**Reported, not gated: the claim boundary.**  The pointwise interval read against the
unsmoothed :math:`\theta(a_0)` at the curvature dose 2.5 is expected to under-cover by
:math:`\Phi(1.96 - b/\mathrm{SE}) - \Phi(-1.96 - b/\mathrm{SE})` with
:math:`b = \theta_h(2.5) - \theta(2.5)` and the efficient SE at n = 4,000: 0.938 at h = 0.5
(the declared bandwidth nearest the exact MSE-optimal 0.60), 0.597 at h = 1.0, and 0.010 at
h = 1.5.  No shared family reads a coverage that must fall below nominal against a second
truth, so the quantity is computed from the binary scenario's primary rows by the design test's
formula and reported, not gated.

**Sizes** (declared before any run, from :mod:`tests.studies.dose_response_pilot`; the design
test recomputes each).

* :data:`CURVE_REPLICATES` = 8,000: the shared X32b rule
  :func:`~tests.studies.repeated_sizing.calibration_size` on the calibration pair, with the C4
  band as its ``joint_coverage=`` term, at the whole calibration share
  :data:`~tests.studies.repeated_sizing.SIZED_FAIL`.  It reads the pooled pilot SE ratio
  1.023 over 8,000 fits of the pair (the single-pair labels ``calibration_4k`` and
  ``calibration_4k_b`` and the pair in ``binary_4k``), the pair's pilot standardized bias, and
  the one-sided 80% lower bound (0.935) of the pilot's joint band coverage, 942 of 1,000.  The
  rule prices the pilot noise of the ratio, which binds: the reported SE is conservative by a
  few percent at n = 4,000, and the calibration band ends at 1.07.  Its failure bound is 0.0160
  at 8,000 against the share 0.0163, and 0.0205 at 7,600.  The efficiency conjunct fails with
  probability below 1e-4 at this size, and each derived control below the discrimination
  floor's 2.5e-4.  Reading the cell off the band fits costs no fit: separate single-pair fits
  would add 8,000 fits of 3.75 s, 8.3 CPU-hours (24.5 CPU-hours for both families instead of
  16.2), and change no row.
* :data:`ROBUSTNESS_REPLICATES` = 800: the positive's pass probability at the one-sided 80%
  upper bound (0.059) of its pilot standardized bias first exceeds
  :data:`~tests.studies.repeated_sizing.SIZED_SHARE` at 800 on a grid of 200; the control's
  pilot bias is 2.87 SDs, discriminated with probability above the floor at any of these sizes.
  At 800 the targeting gain fails its floor with probability below 1e-9 at the pilot
  (``dr_targeting_4k``, 400 draws), so the property instrument's union bound holds.

A replication that raises is never redrawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.dose_response import DoseResponse, pair_name
from tests.parallel import STUDY_JOBS
from tests.studies import dose_response as study
from tests.studies import dose_response_law as law
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.failures import filled
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    paired,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
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
#: The band fits, each of which also gives the calibration cell its row.
CURVE_REPLICATES = 8_000
ROBUSTNESS_REPLICATES = 800
CALIBRATION_N = PROPERTY_N

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

BAND_LABEL = "binary_curve"
CALIBRATION_LABEL = "interior_h1_a2_5"
CALIBRATION_PAIR = (1.0, 2.5)
ROBUSTNESS_LABEL = "h1_a3_5"
ROBUSTNESS_PAIR = (1.0, 3.5)
CALIBRATION_LABELS = (CALIBRATION_LABEL,)
CALIBRATION_CELL = f"{CALIBRATION_LABEL}__correctly_specified"


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
#: The exact efficient SE of the robustness pair at n = 4,000: the targeting gain's unit.
ROBUSTNESS_SE = law.efficiency_sd(*ROBUSTNESS_PAIR) / float(np.sqrt(PROPERTY_N))
#: The least targeting gain, in :data:`ROBUSTNESS_SE` units, that each of the two gain means
#: must reach (the module docstring states the rule and the design test its pass probability).
TARGETING_GAIN_FLOOR = 1.5


@dataclass(frozen=True)
class RobustnessCell:
    """One double-robustness fit.

    Parameters
    ----------
    role : str
        ``"positive"``, ``"control"`` or the diagnostic role of the targeting rows.
    outcome_correct : bool
        Whether the outcome regression is the declared sieve, else the dose-only spline.
    density_declared : bool
        Whether the density is the declared hazard classifier, else the binned marginal one.
    """

    role: str
    outcome_correct: bool
    density_declared: bool


#: The targeted row of the outcome-wrong, declared-density fit, and its plug-in row.
TARGETING_CELL = f"{ROBUSTNESS_LABEL}__outcome_wrong_density_declared"
PLUG_IN_CELL = f"{TARGETING_CELL}_plug_in"

ROBUSTNESS_CELLS: dict[str, RobustnessCell] = {
    f"{ROBUSTNESS_LABEL}__outcome_correct_density_wrong": RobustnessCell(
        "positive", outcome_correct=True, density_declared=False
    ),
    f"{ROBUSTNESS_LABEL}__both_wrong": RobustnessCell(
        "control", outcome_correct=False, density_declared=False
    ),
    TARGETING_CELL: RobustnessCell(DIAGNOSTIC_ROLE, outcome_correct=False, density_declared=True),
}


def robustness_learners(cell: str) -> dict[str, Any]:
    """The learners a robustness fit replaces: its outcome regression and its density."""
    spec = ROBUSTNESS_CELLS[cell]
    form = "full" if spec.density_declared else "marginal"
    return {
        "outcome_learner": study.outcome_learner(interactions=spec.outcome_correct),
        "treatment_learner": study.hazard_learner(len(study.COVARIATES), form=form),
    }


def robustness_fit(frame: pd.DataFrame, cell: str) -> tuple[Any, float]:
    """The fit of one robustness configuration, and its pair's untargeted plug-in."""
    result, initial = study.fit_with_initials(
        frame, study.BINARY, curve=single_pair(*ROBUSTNESS_PAIR), **robustness_learners(cell)
    )
    return result, initial[ROBUSTNESS_NAME]


def calibration_fit(frame: pd.DataFrame) -> Any:
    """The declared binary configuration at the calibration pair alone (the pilot's fit)."""
    return study.fit(frame, study.BINARY, curve=single_pair(*CALIBRATION_PAIR))


def band_fit(frame: pd.DataFrame) -> Any:
    """The fit of the band and calibration cells: the declared binary scenario."""
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

#: ``(family, label, replicate, requested)``: the band label, or a robustness fit.
_Key = tuple[str, str, int, int]


def _budget(declared: int, budget: int | None) -> int:
    return declared if budget is None else min(budget, declared)


def _keys(budget: int | None) -> list[_Key]:
    curve = _budget(CURVE_REPLICATES, budget)
    robust = _budget(ROBUSTNESS_REPLICATES, budget)
    return [
        *((SIMULTANEOUS, BAND_LABEL, r, curve) for r in range(curve)),
        *((ROBUSTNESS, cell, r, robust) for cell in ROBUSTNESS_CELLS for r in range(robust)),
    ]


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, _ = key
    # Every robustness fit of one replicate reads one draw, so the configurations differ by
    # their learners alone.
    stream = ROBUSTNESS_LABEL if family == ROBUSTNESS else label
    return law.sample(PROPERTY_N, stream_seed(STUDY, "property_sample", family, stream, replicate))


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, requested = key
    alpha = STUDY.margins.alpha
    if family == SIMULTANEOUS:
        result = band_fit(frame)
        names = study.ESTIMANDS[study.BINARY]
        return [
            *joint_coverage_rows(
                _Banded(result),
                study.TRUTH[study.BINARY],
                names,
                label=label,
                replicate=replicate,
                n=PROPERTY_N,
                requested=requested,
                pointwise_critical=CRITICAL,
            ),
            replicate_row(
                property_name=CALIBRATION,
                cell=CALIBRATION_CELL,
                role="positive",
                replicate=replicate,
                n=CALIBRATION_N,
                requested=requested,
                truth=CALIBRATION_TRUTH[CALIBRATION_LABEL],
                estimate=result[CALIBRATION_NAME],
                alpha=alpha,
            ),
        ]
    result, plug_in = robustness_fit(frame, label)
    estimate = result[ROBUSTNESS_NAME]
    rows = [
        replicate_row(
            property_name=ROBUSTNESS,
            cell=label,
            role=ROBUSTNESS_CELLS[label].role,
            replicate=replicate,
            n=PROPERTY_N,
            requested=requested,
            truth=ROBUSTNESS_TRUTH,
            estimate=estimate,
            alpha=alpha,
        )
    ]
    if label == TARGETING_CELL:
        # The plug-in is a number, not a fit: it carries the targeted fit's SE.
        rows.append(
            control_row(
                property_name=ROBUSTNESS,
                cell=PLUG_IN_CELL,
                replicate=replicate,
                n=PROPERTY_N,
                requested=requested,
                truth=ROBUSTNESS_TRUTH,
                estimate=float(plug_in),
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
                role=DIAGNOSTIC_ROLE,
            )
        )
    return rows


def _cells_of(key: _Key) -> list[str]:
    family, label, _, _ = key
    if family == SIMULTANEOUS:
        calibration = f"{CALIBRATION}/{CALIBRATION_CELL}"
        return [
            *(f"{SIMULTANEOUS}/{cell}" for cell in joint_cells(label)),
            calibration,
            *DERIVED_CELLS[calibration],
        ]
    if label == TARGETING_CELL:
        return [f"{ROBUSTNESS}/{TARGETING_CELL}", f"{ROBUSTNESS}/{PLUG_IN_CELL}"]
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
        (ROBUSTNESS, cell, ROBUSTNESS_NAME, ROBUSTNESS_TRUTH)
        for cell in (*ROBUSTNESS_CELLS, PLUG_IN_CELL)
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


# ---------------------------------------------------------------------------- targeting gain

#: The two gain means the joint clause reads, by summary column: ``+1`` reads
#: ``plug-in - targeted`` and ``-1`` reads ``plug-in + targeted - 2 truth``, each per replicate
#: in :data:`ROBUSTNESS_SE` units.
TARGETING_COLUMNS: dict[str, int] = {"targeting_gain": 1, "targeting_gain_past_truth": -1}


def targeting_terms(rows: pd.DataFrame, sign: int) -> np.ndarray:
    """The per-replicate gain term of the targeting rows, on the replicates both rows kept.

    Parameters
    ----------
    rows : pandas.DataFrame
        Replication rows that hold the targeted and plug-in cells.
    sign : int
        ``1`` for the gain toward the truth, ``-1`` for the gain past it.

    Returns
    -------
    numpy.ndarray
        ``(plug-in - truth) - sign * (targeted - truth)`` over :data:`ROBUSTNESS_SE`.
    """
    family = rows.loc[rows["property"] == ROBUSTNESS]
    targeted, plug = paired(
        family.loc[family["cell"] == TARGETING_CELL], family.loc[family["cell"] == PLUG_IN_CELL]
    )
    targeted = targeted.sort_values("replicate")
    plug = plug.sort_values("replicate")
    error = {
        name: arm["estimate"].to_numpy(dtype=float) - arm["truth"].to_numpy(dtype=float)
        for name, arm in (("targeted", targeted), ("plug", plug))
    }
    return np.asarray((error["plug"] - sign * error["targeted"]) / ROBUSTNESS_SE)


def targeting_gain_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Gate the targeting gain on the family's joint clause, and read the diagnostic rows.

    Each :data:`TARGETING_COLUMNS` mean must reach :data:`TARGETING_GAIN_FLOOR`; the clause is
    ANDed into the positive's and the control's ``property_passed``.  A diagnostic row is never
    red: its ``passed`` and ``property_passed`` are true, as in the shared band rule, and its
    bias columns are read only through the gain clause.  The declared run at ``dfd56920`` wrote
    the union-model endpoint into ``passed`` instead, which the driver gated, contrary to this
    declaration.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary, written in place.
    rows : pandas.DataFrame
        The replication rows.
    """
    family = summary["property"] == ROBUSTNESS
    if not family.any():
        return
    means = {
        column: float(np.mean(targeting_terms(rows, sign)))
        for column, sign in TARGETING_COLUMNS.items()
    }
    for column, value in means.items():
        summary.loc[family, column] = value
    gained = all(value >= TARGETING_GAIN_FLOOR for value in means.values())
    diagnostic = family & (summary["role"] == DIAGNOSTIC_ROLE)
    summary.loc[diagnostic, "passed"] = True
    summary.loc[diagnostic, "property_passed"] = True
    gated = family & summary["role"].isin(("positive", "control"))
    summary.loc[gated, "property_passed"] = summary.loc[gated, "passed"].astype(bool) & gained


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration and band rules, and the targeting gain."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=tuple(TARGETING_COLUMNS),
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    targeting_gain_verdicts(summary, rows)
    return finish(summary, rates)


def _gain_clause(column: str, sign: int) -> pf.Clause:
    """S10: one gain mean at or above the floor, with the lost replicates at the survivor min.

    The term is a mean of per-replicate values on a fixed scale, so the fill is exact.  Both
    targeting rows come from one fit, so a lost fit costs both the same replicate.
    """

    def judge(mean: float) -> pf.Judged:
        return pf.Judged(mean - TARGETING_GAIN_FLOOR, bool(mean >= TARGETING_GAIN_FLOOR))

    def read(ctx: pf.Context, row: pd.Series) -> pf.Judged:
        return judge(float(row[column]))

    def reads(ctx: pf.Context, row: pd.Series) -> list[tuple[str, str]]:
        return [(ROBUSTNESS, TARGETING_CELL), (ROBUSTNESS, PLUG_IN_CELL)]

    def candidates(ctx: pf.Context, row: pd.Series) -> list[tuple[str, pf.Judged]]:
        values = targeting_terms(ctx.rows, sign)
        lost = max(ctx.lost_in(ROBUSTNESS, TARGETING_CELL), ctx.lost_in(ROBUSTNESS, PLUG_IN_CELL))
        mean = float(np.mean(filled(values, lost, float(np.min(values)))))
        return [("lost rows at survivor min", judge(mean))]

    return pf.Clause(
        column.replace("_", " "),
        "S10",
        pf.roles("positive", "control"),
        read,
        candidates,
        target="property_passed",
        reads=reads,
    )


def _robustness_rule() -> pf.FamilyRule:
    """The shared union-model rule, with the targeting gain as its joint clause."""
    shared = pf.rule_for(ROBUSTNESS)
    assert shared is not None
    gain: tuple[pf.Clause, ...] = tuple(
        _gain_clause(column, sign) for column, sign in TARGETING_COLUMNS.items()
    )
    return pf.FamilyRule(
        families=frozenset({ROBUSTNESS}),
        shared="targeting_gain_verdicts",
        clauses=(*shared.clauses, *gain),
    )


#: The failure accounting's declared rules.  The band family reads the shared rule.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    _robustness_rule(),
)
