r"""Calibration, targeting necessity and the band of ``continuous-survival-functionals`` (X6 (c)).

Fixed before any run (X6 plan rev 2, section 5.4).  Every fit is the primary configuration of
:mod:`tests.studies.continuous_survival_functionals` on the base law at :math:`n = 1{,}000`.

**Calibration** (:data:`CALIBRATION_REPLICATES`, on the stream
``stream_seed(STUDY, "property_sample", family, label, r)``).  The positive cell
``rmst_rd__correctly_specified`` reads the RMST difference at :math:`\tau = 2.32`, the new
functional whose interval no other study reads, against the shared calibration band (SE ratio's
99% interval inside (0.93, 1.07), coverage's inside (0.92, 0.98)).  Its two derived controls are
the shared ones: ``shrunken_se_control`` multiplies each standard error by
:data:`SHRUNKEN_SE_FACTOR` (0.85, the plan's shrunk-SE control) and must leave the band, and
``noise_control`` adds one efficient standard deviation of noise
(:func:`~tests.studies.continuous_survival_functionals_law.rmst_difference_efficiency_sd`, used
only to size it: the cell claims no efficiency).

**Targeting necessity** (reported, no verdict; :data:`NECESSITY_REPLICATES` of the primary
cell's own draws, replicates 0 to 999).  The same fit with the walk targeting the risks only:
:func:`risks_targeted_only` runs the shipped walk on the risk rows and then evaluates the
restricted rows at the hazards it reached, which is what a fit without ``restricted_horizons``
leaves an RMTL at.  Declared reading: that plug-in RMTL's standardized score
:math:`|P_n D_{\mathrm{RMTL}}| / \hat\sigma` exceeds the walk's cutoff in most fits, and its bias
and coverage are published beside the targeted fit's.  If its bias is inside the 0.2 SD margin,
the reading says this law does not separate the two, and the claim that the RMTL rows must be
targeted rests on the score test of ``tests/unit/test_continuous_survival_functionals.py``, not
on this cell.  The two artifacts are ``necessity-cells.csv`` and ``necessity-replicates.csv``.

**Band, second phase** (:data:`BAND_PHASE`).  The joint sup-t band over every risk and risk
difference of the fit, read for joint coverage by ``joint_band_pass_probability`` (0.998 at
2,000), with the pointwise-only control (critical value 1.96) below the joint floor.  It waits
for the band thread's PR 2, so the band measured is the default one, and runs as a second phase
of this study under its own declaration.  This module does not fit it.

A replication that raises is never redrawn.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.survival import _targeting
from cleverly.survival import engine as engine_module
from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import recordable
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_functionals as study
from tests.studies import continuous_survival_functionals_law as functionals
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
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

CALIBRATION = "interval_calibration"
CALIBRATION_N = study.PRIMARY_N
#: The plan's design value (5.4): a calibration row passes with probability 0.997 at an SE
#: ratio of 1.00 and the shrunk-SE control is detected with probability 1.000.
CALIBRATION_REPLICATES = 3_200
SHRUNKEN_SE_FACTOR = 0.85
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

RMST_RD = functionals.rmst_difference_name()
CALIBRATION_LABELS = ("rmst_rd",)
EFFICIENCY_SD = {"rmst_rd": functionals.rmst_difference_efficiency_sd()}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

NECESSITY_REPLICATES = 1_000
#: The RMTL targets the necessity reading publishes, each with its walk index.
NECESSITY_TARGETS: tuple[str, ...] = tuple(
    functionals.rmtl_name(arm, cause) for arm in law.ARMS for cause in law.CAUSES.values()
)
CELLS_FILE = "necessity-cells.csv"
REPLICATES_FILE = "necessity-replicates.csv"

BAND_PHASE = "second phase, after the band thread's PR 2"


# ---------------------------------------------------------------- the risk-only walk


@contextmanager
def risks_targeted_only(risk_rows: int) -> Iterator[None]:
    """Run the engine's walk on its first ``risk_rows`` target rows, then evaluate them all.

    Parameters
    ----------
    risk_rows : int
        The number of risk rows per arm and cause (the target times); the restricted rows
        follow them in :attr:`~cleverly.survival._targeting.Problem.target_weights`.

    Yields
    ------
    None
        While open, a fit's restricted means are the plug-ins at the risk-targeted hazards,
        and its record's standardized scores include their scores there.
    """
    original = engine_module.walk

    def walk(problem: Any, dlam: Any, **options: Any) -> Any:
        initial = np.array(dlam, copy=True)
        risks = replace(problem, target_weights=problem.target_weights[:risk_rows])
        result = original(risks, dlam, **options)
        return replace(
            result,
            evaluation=_targeting.evaluate(problem, result.dlam),
            initial=_targeting.evaluate(problem, initial),
        )

    engine_module.walk = walk
    try:
        yield
    finally:
        engine_module.walk = original


def _walk_index(arm: str, cause: str) -> int:
    """The walk's index of an arm's RMTL of one cause: ``(a J + j) Q + K``."""
    arms = (law.REFERENCE, next(a for a in law.ARMS if a != law.REFERENCE))
    causes = tuple(law.CAUSES.values())
    K = len(law.TIMES)
    return (arms.index(arm) * len(causes) + causes.index(cause)) * (K + 1) + K


# ---------------------------------------------------------------------------- keys

#: ``(family, label, replicate, requested)``.
_Key = tuple[str, str, int, int]


def _keys(budget: int | None) -> list[_Key]:
    count = CALIBRATION_REPLICATES if budget is None else min(budget, CALIBRATION_REPLICATES)
    return [(CALIBRATION, label, r, count) for label in CALIBRATION_LABELS for r in range(count)]


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, _ = key
    return registered.sample(
        CALIBRATION_N, stream_seed(STUDY, "property_sample", family, label, replicate)
    )


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, label, replicate, requested = key
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate = study.fit(frame)[RMST_RD]
    return [
        replicate_row(
            property_name=CALIBRATION,
            cell=f"{label}__correctly_specified",
            role="positive",
            replicate=replicate,
            n=CALIBRATION_N,
            requested=requested,
            truth=functionals.TRUTH[RMST_RD],
            estimate=estimate,
            alpha=STUDY.margins.alpha,
        )
    ]


def _cells_of(key: _Key) -> list[str]:
    name = f"{CALIBRATION}/{key[1]}__correctly_specified"
    return [name, *DERIVED_CELLS[name]]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads: the base law, with the study's truths."""

    name: str = "base"

    def truth(self) -> dict[str, float]:
        """The study's truths."""
        return dict(functionals.TRUTH)


def declared_cells() -> tuple[PropertyCell, ...]:
    """The sampled calibration cell; its two controls are derived from its rows."""
    return tuple(
        PropertyCell(
            property=CALIBRATION,
            cell=f"{label}__correctly_specified",
            dgp=DeclaredLaw(),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES,
            seed=stream_seed(STUDY, "property_sample", CALIBRATION, label, 0),
            role="positive",
            estimand=RMST_RD,
        )
        for label in CALIBRATION_LABELS
    )


def synthetic_scale(cell: PropertyCell) -> float:
    """The efficient standard deviation of the cell's estimate over :math:`\\sqrt n`."""
    return EFFICIENCY_SD[cell.cell.split("__")[0]] / float(np.sqrt(cell.n))


def declared_truths() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries."""
    return tuple(
        (CALIBRATION, f"{label}__{kind}", RMST_RD, functionals.TRUTH[RMST_RD])
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    )


#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every calibration replication; ``budget`` caps the cell for a smoke run."""
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
    """The calibration rule with its two controls; no efficiency claim."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    return finish(summary, rates)


#: The failure accounting's declared rule.
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None),)


# ------------------------------------------------------------- targeting necessity


def _necessity(replicate: int) -> list[dict[str, Any]]:
    frame, truth = study.draw_scenario(study.GATED_CELL, study.PRIMARY_N, replicate)
    base = {"replicate": replicate, "n": study.PRIMARY_N}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with risks_targeted_only(len(law.TIMES)):
                result = study.fit(frame)
    except Exception as error:
        if not recordable(error):
            raise
        return [
            {
                **base,
                "estimand": name,
                "truth": truth[name],
                "estimate": np.nan,
                "std_error": np.nan,
                "covered": np.nan,
                "standardized_score": np.nan,
                "cutoff": np.nan,
                "failure": type(error).__name__,
            }
            for name in NECESSITY_TARGETS
        ]
    rows = []
    scores = result.record.standardized_score
    for arm in law.ARMS:
        for cause in law.CAUSES.values():
            name = functionals.rmtl_name(arm, cause)
            estimate = result[name]
            low, high = estimate.ci
            rows.append(
                {
                    **base,
                    "estimand": name,
                    "truth": truth[name],
                    "estimate": float(estimate.psi),
                    "std_error": float(estimate.std_error),
                    "covered": float(low <= truth[name] <= high),
                    "standardized_score": float(scores[_walk_index(arm, cause)]),
                    "cutoff": float(result.record.cutoff),
                    "failure": "",
                }
            )
    return rows


def necessity_rows(*, budget: int | None = None, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """The risk-only fits' RMTL rows, in a fixed order."""
    count = NECESSITY_REPLICATES if budget is None else min(budget, NECESSITY_REPLICATES)
    outcomes = map_parallel(_necessity, list(range(count)), n_jobs=n_jobs)
    rows = pd.DataFrame([row for rows in outcomes for row in rows])
    return rows.sort_values(["estimand", "replicate"], ignore_index=True)


def summarize_necessity(rows: pd.DataFrame) -> pd.DataFrame:
    """Per RMTL: fits, failures, bias, spread, standardized bias, coverage, share above cutoff."""
    records = []
    for estimand, group in rows.groupby("estimand", sort=True):
        kept = group.loc[group["failure"].fillna("").eq("")]
        error = kept["estimate"] - kept["truth"]
        spread = float(error.std(ddof=1))
        records.append(
            {
                "estimand": estimand,
                "n": int(group["n"].iloc[0]),
                "fits": len(kept),
                "failures": len(group) - len(kept),
                "bias": float(error.mean()),
                "empirical_sd": spread,
                "standardized_bias": float(error.mean() / spread),
                "se_ratio": float(kept["std_error"].mean() / spread),
                "coverage": float(kept["covered"].mean()),
                "share_above_cutoff": float((kept["standardized_score"] > kept["cutoff"]).mean()),
            }
        )
    return pd.DataFrame(records)


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The calibration rows and the targeting-necessity reading's two artifacts."""
    rows = necessity_rows(budget=budget, n_jobs=n_jobs)
    return (
        generate_property_rows(n_jobs=n_jobs, budget=budget),
        {CELLS_FILE: summarize_necessity(rows), REPLICATES_FILE: rows},
    )
