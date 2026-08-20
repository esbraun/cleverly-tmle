"""Independent repeated-sampling properties for ordinary canonical TMLE.

The claims are van der Laan and Rubin's: double robustness, root-n behaviour with local
efficiency, and calibrated inference.  Each cell here is a law and a nuisance configuration
chosen so that a claim can *fail*; the verdicts come from
:mod:`tests.studies.evidence.properties` and are bounded by the same margins the
implementation comparison declares, so the two halves of the study cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import DGP, linear_dgp, nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.utils.bounds import expit
from tests.conftest import OracleOutcomeContinuous, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_tmle import G_BOUNDS, STUDY
from tests.studies.evidence.properties import (
    PropertyCell,
    rate,
    run_cells,
    summarize_cells,
)

#: Sized by what the claim needs, not by habit.  The 99% interval's half-width is about
#: ``2.6 / sqrt(m)`` empirical standard deviations, so ``m`` decides the largest bias the study
#: can still place inside the 0.25 margin: 300 replications resolve 0.10, and 1,200 resolve
#: 0.176.  The "only the treatment mechanism correct" cell really does carry about 0.14 of a
#: standard deviation at ``n = 700`` -- it is the arm of double robustness that leans on the
#: inverse-weighting term -- so at 300 the study could not say whether that was inside the
#: margin, and said so.  Raising the budget cannot buy a pass for a bad estimator: the interval
#: contracts on the truth, so a bias past the margin becomes *discriminated* instead.
DOUBLE_ROBUST_REPLICATES = 1_200
RATE_REPLICATES = 200
NULL_REPLICATES = 400

#: Three sizes, not two: a rate estimated from two points is a ratio with no residual, and
#: quadrupling twice separates a root-n contraction from a merely decreasing one.
RATE_SIZES = (500, 2000, 8000)

#: The expected contraction rate of the sampling distribution, and the alternative the
#: interval must exclude for the check to have discriminated anything.
ROOT_N_SLOPE = -0.5
EXCLUDED_SLOPE = -0.25

#: Power positive control: an effect this large is rejected essentially always, so a
#: rejection indicator that never fires cannot pass the type-I cell by being inert.
ALTERNATIVE_EFFECT = 0.5
MINIMUM_POWER = 0.80


def _null_dgp(effect: float = 0.0) -> DGP:
    """Confounded, with a treatment effect of ``effect``.

    The sharp null keeps the confounding: an unadjusted comparison of arms is biased here, so
    a calibrated rejection rate is evidence about the estimator rather than about a
    randomized experiment.
    """

    def propensity(w: np.ndarray) -> np.ndarray:
        return expit(0.5 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        return 1.0 + 0.9 * w[:, 0] + 0.6 * w[:, 1] - 0.4 * w[:, 2] + effect * a

    return DGP(
        name=f"canonical_tmle_null_{effect:g}",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def cells() -> tuple[PropertyCell, ...]:
    """Every property cell, with the seed each is drawn with.

    Distinct seeds throughout.  The sizes in the rate study in particular are meant to be
    independent runs -- the slope's interval is computed as though they were -- and sharing a
    seed across them would make that assumption false for no gain.
    """
    nonlinear = nonlinear_dgp()
    linear = linear_dgp()
    nuisances: tuple[tuple[str, Callable[[], Any], Callable[[], Any]], ...] = (
        (
            "both_correct",
            lambda: OracleOutcomeContinuous(nonlinear),
            lambda: OracleTreatment(nonlinear),
        ),
        (
            "outcome_correct",
            lambda: OracleOutcomeContinuous(nonlinear),
            lambda: LogisticRegression(max_iter=1000),
        ),
        ("treatment_correct", lambda: LinearRegression(), lambda: OracleTreatment(nonlinear)),
        ("both_wrong", lambda: LinearRegression(), lambda: LogisticRegression(max_iter=1000)),
    )
    out: list[PropertyCell] = []
    for index, (name, outcome, treatment) in enumerate(nuisances):
        out.append(
            PropertyCell(
                property="double_robustness",
                cell=name,
                dgp=nonlinear,
                outcome_learner=outcome,
                treatment_learner=treatment,
                n=700,
                replicates=DOUBLE_ROBUST_REPLICATES,
                seed=7100 + index,
            )
        )
    for index, size in enumerate(RATE_SIZES):
        out.append(
            PropertyCell(
                property="root_n_and_efficiency",
                cell=f"n_{size}",
                dgp=linear,
                outcome_learner=LinearRegression,
                treatment_learner=lambda: LogisticRegression(max_iter=1000),
                n=size,
                replicates=RATE_REPLICATES,
                seed=8100 + 100 * index,
            )
        )
    out.append(
        PropertyCell(
            property="type_i_error",
            cell="sharp_null",
            dgp=_null_dgp(),
            outcome_learner=LinearRegression,
            treatment_learner=lambda: LogisticRegression(max_iter=1000),
            n=1000,
            replicates=NULL_REPLICATES,
            seed=9100,
        )
    )
    out.append(
        PropertyCell(
            property="power",
            cell="alternative",
            dgp=_null_dgp(ALTERNATIVE_EFFECT),
            outcome_learner=LinearRegression,
            treatment_learner=lambda: LogisticRegression(max_iter=1000),
            n=1000,
            replicates=NULL_REPLICATES,
            seed=9200,
        )
    )
    return tuple(out)


def _estimator(cell: PropertyCell) -> Callable[[], Any]:
    return lambda: TMLE(
        outcome_learner=cell.outcome_learner(),
        treatment_learner=cell.treatment_learner(),
        cross_fit=False,
        # A bare name is ``resolve_estimands``'s single-estimand form, and unlike a
        # one-tuple it does not have to claim a narrower element type than the cell has.
        estimands=cell.estimand,
        simultaneous=False,
        g_bounds=G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Run every property cell and return the per-replication rows."""
    return run_cells(cells(), _estimator, n_jobs=n_jobs)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-cell descriptive summary, the two rate rows, and every verdict.

    Each verdict is an interval statement against a margin declared before the run.  For the
    double-robustness cells that means the positive cells must establish that the bias is
    *inside* the margin and the both-wrong control must establish that it is *outside* it --
    the same instrument in both directions, so neither can be passed by a study too small to
    say anything.
    """
    margins = STUDY.margins
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
    )
    summary["slope"] = np.nan
    summary["slope_ci_lower"] = np.nan
    summary["slope_ci_upper"] = np.nan
    summary["passed"] = False

    positive = (summary["property"] == "double_robustness") & (summary["cell"] != "both_wrong")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    control = (summary["property"] == "double_robustness") & (summary["cell"] == "both_wrong")
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    efficiency = summary["property"] == "root_n_and_efficiency"
    summary.loc[efficiency, "passed"] = (
        (summary.loc[efficiency, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[efficiency, "se_ratio"].between(*margins.se_ratio_sanity)
        & summary.loc[efficiency, "bias_equivalent"]
    )

    null = summary["property"] == "type_i_error"
    # One-sided: a test that over-rejects is invalid, and one that under-rejects is
    # conservative.  The power cell below is what stops "never rejects" from passing here.
    summary.loc[null, "passed"] = (
        summary.loc[null, "rejection_ci_upper"] <= margins.alpha + margins.type_i_margin
    ) & (summary.loc[null, "coverage_ci_lower"] >= margins.coverage_floor)

    power = summary["property"] == "power"
    summary.loc[power, "passed"] = summary.loc[power, "rejection_ci_lower"] >= MINIMUM_POWER

    rates: list[dict[str, Any]] = []
    for statistic, cell in (("spread", "empirical_sd"), ("reported", "reported_se")):
        fitted = rate(
            rows,
            property_name="root_n_and_efficiency",
            statistic=statistic,
            bootstrap_replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=STUDY.seed + 30_000 + len(rates),
        )
        row: dict[str, Any] = dict.fromkeys(summary.columns, np.nan)
        row.update(
            {
                "property": "root_n_rate",
                "cell": cell,
                "n": max(RATE_SIZES),
                "replicates": RATE_REPLICATES * len(RATE_SIZES),
                "failed_replicates": 0,
                "slope": fitted.slope,
                "slope_ci_lower": fitted.interval.low,
                "slope_ci_upper": fitted.interval.high,
                "passed": bool(
                    fitted.consistent_with(ROOT_N_SLOPE) and fitted.excludes(EXCLUDED_SLOPE)
                ),
            }
        )
        rates.append(row)
    summary = pd.concat([summary, pd.DataFrame(rates)], ignore_index=True)
    summary["passed"] = summary["passed"].astype(bool)
    return summary.sort_values(["property", "cell"], ignore_index=True)
