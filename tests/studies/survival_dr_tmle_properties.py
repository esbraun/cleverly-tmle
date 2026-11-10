r"""Repeated-sampling properties of the drift-corrected survival TMLE, and its reported cells.

Two families, fixed before any run, on law L-S (:mod:`tests.studies.survival_dr_tmle_law`),
cross-fitted with the study's 10 folds.  "Wrong" for the hazard is the logistic regression on
``W1``, ``A`` and time; for the treatment it is the logistic regression on ``W1`` alone; for
censoring it is a constant retention (``DummyClassifier(strategy="prior")``).

**Double robustness** (:data:`ROBUSTNESS_REPLICATES` each, :math:`n = 16{,}000`).

=====================================  ========  ============================================
cell                                   role      configuration
=====================================  ========  ============================================
``a1_t5__mechanism_correct``           positive  hazard wrong; treatment and censoring
                                                 correct (the design's L-param, row 4)
``a1_t5__outcome_correct``             positive  hazard correct; treatment and censoring
                                                 wrong
``a1_t5__both_wrong``                  control   all three wrong
=====================================  ========  ============================================

Each reads the risk of arm 1 at :math:`t = 5`.  The control's gap is read from the pilot and
stated in ``tests/canonical/survival_dr_tmle/regenerate.py`` before the run.

**Calibration** (:data:`CALIBRATION_REPLICATES`, :math:`n = 16{,}000`).  The positive cell
``both_rd__correctly_specified`` fits the record's ``both`` configuration and reads the risk
difference at :math:`t = 5`; its SE ratio is read against the shared calibration band and its
efficiency ratio against :data:`EFFICIENCY_RATIO_BAND`, with the exact efficient spread of
:func:`~tests.studies.survival_dr_tmle_law.efficiency_sd`.  The two shared controls are derived
from its rows.

**Reported cells** (no verdict; :data:`REPORTED_REPLICATES` each, on the stream
``stream_seed(STUDY, "reported", cell, r)``).  ``insample_4k`` fits the hazard model wrong and
gradient boosting for the treatment and censoring in sample (``n_folds=1``), Theorem 2 without
its condition C2.  ``slow_censoring_<n>`` is the design's L-trial-slow: the trial law, the
treatment declared, the hazard model wrong and the censoring model correct plus the logit offset
:math:`c\,n^{-1/3} W_2`; ``slow_hazard_<n>`` is L-hslow: the hazard model correct plus the same
offset, the treatment on ``W1`` and constant censoring.  Both slow cells run at the record's
three sizes with :data:`SLOW_OFFSET`; their standardized bias is expected to grow with
:math:`n`.  They publish as ``reported-cells.csv`` and ``reported-replicates.csv``.

A replication that raises is never redrawn.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from cleverly.longitudinal import SurvivalDRTMLE
from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import recordable
from tests.parallel import STUDY_JOBS
from tests.studies import survival_dr_tmle as study
from tests.studies import survival_dr_tmle_law as law
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
from tests.studies.stratified_incremental_msm_properties import Columns

STUDY = study.STUDY

ROBUSTNESS = "double_robustness"
CALIBRATION = "interval_calibration"

ROBUSTNESS_N = 16_000
ROBUSTNESS_REPLICATES = 1_000
CALIBRATION_N = 16_000
CALIBRATION_REPLICATES = 4_800
REPORTED_REPLICATES = 400

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The constant :math:`c` of the slow cells' logit offset :math:`c\,n^{-1/3} W_2`.
SLOW_OFFSET = 4.0

ARM1 = law.ESTIMANDS[0]
RD = law.ESTIMANDS[2]

#: ``cell -> (configuration, role)``.
ROBUSTNESS_CELLS: dict[str, tuple[str, str]] = {
    "a1_t5__mechanism_correct": ("mechanism_correct", "positive"),
    "a1_t5__outcome_correct": ("outcome_correct", "positive"),
    "a1_t5__both_wrong": ("both_wrong", "control"),
}
CALIBRATION_LABELS = ("both_rd",)
EFFICIENCY_SD = {"both_rd": law.efficiency_sd(trial=False)[RD]}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

#: ``cell -> (configuration, n)`` of the reported cells.
REPORTED_CELLS: dict[str, tuple[str, int]] = {
    "insample_4k": ("insample", 4_000),
    **{f"slow_censoring_{n // 1000}k": ("slow_censoring", n) for n in study.SIZES},
    **{f"slow_hazard_{n // 1000}k": ("slow_hazard", n) for n in study.SIZES},
}
CELLS_FILE = "reported-cells.csv"
REPLICATES_FILE = "reported-replicates.csv"


# ------------------------------------------------------------------------- learners


class Offset(BaseEstimator, ClassifierMixin):
    """A classifier whose logit is moved by ``shift`` times one design column.

    Parameters
    ----------
    base : estimator
        The classifier fitted on the design.
    column : int
        The design column the shift multiplies.
    shift : float
        The coefficient added to the fitted logit.
    """

    def __init__(self, base: Any = None, column: int = 1, shift: float = 0.0) -> None:
        self.base = base
        self.column = column
        self.shift = shift

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> Offset:
        self.model_ = clone(self.base).fit(design, target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        fitted = np.clip(self.model_.predict_proba(values)[:, 1], 1e-12, 1.0 - 1e-12)
        moved = expit(logit(fitted) + self.shift * values[:, self.column])
        return np.column_stack([1.0 - moved, moved])


def _logistic() -> Any:
    return study._logistic()


def _hazard_wrong() -> Any:
    return Columns(_logistic(), keep=(0, 2, 3))


def _treatment_wrong() -> Any:
    return Columns(_logistic(), keep=(0,))


def _constant() -> Any:
    return DummyClassifier(strategy="prior")


def configuration_settings(configuration: str, n: int) -> tuple[bool, dict[str, Any]]:
    """Whether the configuration draws the trial law, and its fit settings."""
    common = study.settings("both")
    slow = SLOW_OFFSET * n ** (-1.0 / 3.0)
    if configuration == "mechanism_correct":
        return False, {**common, "outcome_learner": _hazard_wrong()}
    if configuration == "outcome_correct":
        return False, {
            **common,
            "treatment_learner": _treatment_wrong(),
            "censoring_learner": _constant(),
        }
    if configuration == "both_wrong":
        return False, {
            **common,
            "outcome_learner": _hazard_wrong(),
            "treatment_learner": _treatment_wrong(),
            "censoring_learner": _constant(),
        }
    if configuration == "insample":
        return False, {
            **common,
            "outcome_learner": _hazard_wrong(),
            "treatment_learner": HistGradientBoostingClassifier(random_state=0),
            "censoring_learner": HistGradientBoostingClassifier(random_state=0),
            "n_folds": 1,
        }
    if configuration == "slow_censoring":
        # The retention design is [W1, W2, A]: the offset moves the retention logit along W2.
        return True, {
            **common,
            "outcome_learner": _hazard_wrong(),
            "censoring_learner": Offset(_logistic(), column=1, shift=slow),
        }
    if configuration == "slow_hazard":
        return False, {
            **common,
            "outcome_learner": Offset(_logistic(), column=1, shift=slow),
            "treatment_learner": _treatment_wrong(),
            "censoring_learner": _constant(),
        }
    raise KeyError(configuration)


def configured_fit(frame: pd.DataFrame, configuration: str, n: int) -> Any:
    """The subject's fit of one property or reported configuration."""
    trial, chosen = configuration_settings(configuration, n)
    # The slow-censoring cell draws the trial law and declares the treatment only.
    data = study.container(frame, treatment=trial)
    return SurvivalDRTMLE(law.ARMS, **chosen).fit(data)


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
    n = ROBUSTNESS_N if family == ROBUSTNESS else CALIBRATION_N
    return law.sample(n, seed, trial=False)


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, label, replicate, requested = key
    alpha = STUDY.margins.alpha
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == ROBUSTNESS:
            configuration, role = ROBUSTNESS_CELLS[label]
            estimate = configured_fit(frame, configuration, ROBUSTNESS_N)[ARM1]
            return [
                replicate_row(
                    property_name=ROBUSTNESS,
                    cell=label,
                    role=role,
                    replicate=replicate,
                    n=ROBUSTNESS_N,
                    requested=requested,
                    truth=law.TRUTH[ARM1],
                    estimate=estimate,
                    alpha=alpha,
                )
            ]
        estimate = study.fit(frame, "both_16k")[RD]
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
    robust = tuple((ROBUSTNESS, cell, ARM1, law.TRUTH[ARM1]) for cell in ROBUSTNESS_CELLS)
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
    configuration, n = REPORTED_CELLS[cell]
    trial, _ = configuration_settings(configuration, n)
    frame = law.sample(n, reported_seed(cell, replicate), trial=trial)
    base = {"scenario": cell, "replicate": replicate, "n": n}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = configured_fit(frame, configuration, n)
    except Exception as error:
        if not recordable(error):
            raise
        return [
            {
                **base,
                "estimand": name,
                "truth": law.TRUTH[name],
                "estimate": np.nan,
                "std_error": np.nan,
                "covered": np.nan,
                "failure": type(error).__name__,
            }
            for name in law.ESTIMANDS
        ]
    rows = []
    for name in law.ESTIMANDS:
        estimate = result[name]
        low, high = estimate.ci
        rows.append(
            {
                **base,
                "estimand": name,
                "truth": law.TRUTH[name],
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "covered": float(low <= law.TRUTH[name] <= high),
                "failure": "",
            }
        )
    return rows


def reported_rows(
    *, replicates: int = REPORTED_REPLICATES, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    """Every reported cell's replicate rows, in a fixed order."""
    jobs = [(cell, r) for cell in REPORTED_CELLS for r in range(replicates)]
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
    """The property rows and the reported cells' two artifacts."""
    rows = reported_rows(
        replicates=REPORTED_REPLICATES if budget is None else min(budget, REPORTED_REPLICATES),
        n_jobs=n_jobs,
    )
    return (
        generate_property_rows(n_jobs=n_jobs, budget=budget),
        {CELLS_FILE: summarize_cells(rows), REPLICATES_FILE: rows},
    )
