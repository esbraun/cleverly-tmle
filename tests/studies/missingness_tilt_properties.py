r"""Repeated-sampling properties of the targeted missingness tilt (A2, CDR-001).

Two families, fixed before any run.  Each fits the in-sample construction of
:mod:`tests.studies.missingness_tilt_inference` at n = 1,000 and reads the tilted ``ate`` at
:math:`\gamma = 2`, the declared tilt farthest from MAR, on its own stream:
``stream_seed(STUDY, "property_sample", <family>, "ate_gamma2", r)``.

**Interval calibration** (``ate_gamma2``, R = :data:`CALIBRATION_REPLICATES`).  The positive is
the targeted tilt's Wald interval with the declared, correctly specified learners.  Three
controls read the same fits:

========================================  ======================================================
cell                                      what it must show
========================================  ======================================================
``ate_gamma2__correctly_specified``       SE ratio in [0.93, 1.07], coverage in [0.92, 0.98], and
                                          both efficiency ratios against :data:`EFFICIENCY_SD`
                                          (the law's exact tilted efficient SD) in
                                          :data:`EFFICIENCY_RATIO_BAND`
``ate_gamma2__shrunken_se_control``       the reported SE times :data:`SHRUNKEN_SE_FACTOR`: its SE
                                          ratio interval falls below the band
``ate_gamma2__noise_control``             one efficiency unit of independent noise on each
                                          estimate: its empirical efficiency ratio rises above
                                          the band
``ate_gamma2__mar_variance_control``      the A1-era interval: the untargeted tilted plug-in with
                                          the fit's MAR standard error of ``ate``.  Its SE ratio
                                          interval must fall below the band
========================================  ======================================================

The A1-era control is a fit-free reading of the same fit, so a lost positive loses it too.  Its
SE ratio is the MAR efficient SD over the tilted plug-in's spread: on the law the MAR efficient
SD of ``ate`` is 0.898 and the tilted one at :math:`\gamma = 2` is 1.105 (ratio 0.81), and the
local pilot reads 0.84 with coverage 0.90 (1,000 draws).  It needs ``mar_variance_control`` in
:data:`~tests.studies.evidence.property_verdicts.SE_CONTROL_SIDES` (side ``below``).

**Targeting necessity** (``ate_gamma2``, R = :data:`NECESSITY_REPLICATES`).  The outcome
regression drops the law's ``A W1`` interaction (:func:`main_terms_outcome_learner`), so it
converges to a projection, not to :math:`\bar Q_0`, while both mechanisms stay correctly
specified.  The positive arm is the targeted tilt, whose first-order bias vanishes with correct
mechanisms and whose remainder is second order in the regression error,
:math:`E[(1 - \pi)\{h(\bar Q) - h(\bar Q_0) - h'(\bar Q)(\bar Q - \bar Q_0)\}]`; the control
arm is the untargeted tilted plug-in on the MAR-targeted regression, whose first-order bias is
:math:`\operatorname{cov}(\kappa, \bar Q^* - \bar Q_0)`, nonzero because the dropped term moves
the regression error with its level.  The positive's bias must be equivalent to zero, the
control's discriminated from it, and the displacement at least
:data:`TARGETING_DISPLACEMENT` positive-arm SDs.  The local pilot reads targeted +0.02 and
untargeted +0.42 SDs (400 draws).  The positive's limit is not exactly the truth: the remainder above is a
fixed nonzero number for a fixed misspecified regression, and the pilot puts it well inside the
margin at n = 1,000.

**Sizes** (declared before any run, from :mod:`tests.studies.missingness_tilt_pilot`).
Provisional: they read the small local pilot (calibration 1,000 draws, pilot SE ratio 1.013;
necessity 400 draws), and the runner's full pilot replaces them.

* :data:`CALIBRATION_REPLICATES`: the smallest multiple of 400 at which
  :func:`~tests.studies.repeated_sizing.calibration_pass` at the one-sided 80% bounds of the
  pilot SE ratio is at least :data:`~tests.studies.repeated_sizing.SIZED_SHARE`, and each
  control discriminates with probability above ``1 - DISCRIMINATION_FAIL``.
* :data:`NECESSITY_REPLICATES`: the smallest multiple of 200 at which the positive's bias
  equivalence at the one-sided 80% upper bound of its pilot bias, times the control's
  discrimination at the lower bound of its pilot bias, is at least ``SIZED_SHARE``.

A replication that raises is never redrawn.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from tests.parallel import STUDY_JOBS
from tests.studies import missingness_tilt_inference as study
from tests.studies import missingness_tilt_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import (
    CALIBRATION_CONTROL_KINDS,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY

CALIBRATION = "interval_calibration"
NECESSITY = "targeting_necessity"
LABEL = "ate_gamma2"
TARGET = "ate"
GAMMA = 2.0
NAME = study.estimand_name(TARGET, GAMMA)

PROPERTY_N = study.PRIMARY_N
CALIBRATION_N = PROPERTY_N
#: Provisional, from the small local pilot (see the module docstring).
CALIBRATION_REPLICATES = 8_400
NECESSITY_REPLICATES = 1_600

SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
TARGETING_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

TRUTH = law.truths(GAMMA)[TARGET]
EFFICIENCY_SD = {LABEL: law.efficiency_sd(GAMMA)}
MAR_CONTROL = "mar_variance_control"
POSITIVE_CELL = f"{LABEL}__correctly_specified"
MAR_CELL = f"{LABEL}__{MAR_CONTROL}"
DERIVED_CELLS = calibration_derived((LABEL,))


def main_terms_outcome_learner() -> Pipeline:
    """The necessity family's outcome regression: main terms, without the ``A W1`` term."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=study.UNPENALIZED_C, max_iter=5000))


def property_fit(frame: pd.DataFrame, family: str) -> Any:
    """The in-sample fit of ``ate`` a family reads."""
    overrides: dict[str, Any] = {"estimands": (TARGET,)}
    if family == NECESSITY:
        overrides["outcome_learner"] = main_terms_outcome_learner()
    return study.fit(frame, study.IN_SAMPLE, **overrides)


def tilted(result: Any) -> tuple[float, float, float]:
    """``(targeted estimate, its SE, untargeted plug-in)`` of the tilted ``ate`` at gamma 2."""
    row = study.tilt(result, (TARGET,), (GAMMA,)).to_frame().iloc[0]
    plug_in = study.initial_estimates(result, (TARGET,), (GAMMA,))[NAME]
    return float(row["psi"]), float(row["std_err"]), plug_in


# ---------------------------------------------------------------------------- keys

#: ``(family, label, replicate, requested)``.
_Key = tuple[str, str, int, int]


def _budget(declared: int, budget: int | None) -> int:
    return declared if budget is None else min(budget, declared)


def _keys(budget: int | None) -> list[_Key]:
    calibration = _budget(CALIBRATION_REPLICATES, budget)
    necessity = _budget(NECESSITY_REPLICATES, budget)
    return [
        *((CALIBRATION, LABEL, r, calibration) for r in range(calibration)),
        *((NECESSITY, LABEL, r, necessity) for r in range(necessity)),
    ]


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, _ = key
    return law.sample(PROPERTY_N, stream_seed(STUDY, "property_sample", family, label, replicate))


def _row(
    family: str, cell: str, key: _Key, estimate: float, std_error: float, role: str
) -> dict[str, Any]:
    _, _, replicate, requested = key
    return control_row(
        property_name=family,
        cell=cell,
        replicate=replicate,
        n=PROPERTY_N,
        requested=requested,
        truth=TRUTH,
        estimate=estimate,
        standard_error=std_error,
        critical=CRITICAL,
        role=role,
    )


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family = key[0]
    result = property_fit(frame, family)
    estimate, std_error, plug_in = tilted(result)
    if family == CALIBRATION:
        return [
            _row(CALIBRATION, POSITIVE_CELL, key, estimate, std_error, "positive"),
            # The A1-era interval: the MAR SE of the fit's own ate, centred on the plug-in.
            _row(CALIBRATION, MAR_CELL, key, plug_in, float(result[TARGET].std_error), "control"),
        ]
    # The plug-in is a number, not a fit: it carries the targeted fit's SE.
    return [
        _row(NECESSITY, f"{LABEL}__targeted", key, estimate, std_error, "positive"),
        _row(NECESSITY, f"{LABEL}__untargeted", key, plug_in, std_error, "control"),
    ]


def _cells_of(key: _Key) -> list[str]:
    family = key[0]
    if family == CALIBRATION:
        positive = f"{CALIBRATION}/{POSITIVE_CELL}"
        return [positive, f"{CALIBRATION}/{MAR_CELL}", *DERIVED_CELLS[positive]]
    return [f"{NECESSITY}/{LABEL}__targeted", f"{NECESSITY}/{LABEL}__untargeted"]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries."""
    calibrated = tuple(
        (CALIBRATION, f"{LABEL}__{kind}", NAME, TRUTH)
        for kind in ("correctly_specified", *CALIBRATION_CONTROL_KINDS, MAR_CONTROL)
    )
    necessary = tuple(
        (NECESSITY, f"{LABEL}__{arm}", NAME, TRUTH) for arm in ("targeted", "untargeted")
    )
    return calibrated + necessary


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
                labels=(LABEL,),
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


#: The ``targeting_necessity`` family's declaration, read by its verdict and its rule.
TARGETING_NECESSITY = property_verdicts.Necessity(
    family=NECESSITY,
    labels=(LABEL,),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration rule and the necessity rule."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement",),
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    return finish(summary, rates)


#: The failure accounting's declared rules.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
