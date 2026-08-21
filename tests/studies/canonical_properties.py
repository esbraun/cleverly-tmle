"""Independent repeated-sampling properties for ordinary canonical TMLE.

The claims are van der Laan and Rubin's: double robustness, root-n behaviour with local
efficiency, and calibrated inference.  Each cell here is a law and a nuisance configuration
chosen so that a claim can *fail*; the verdicts come from
:mod:`tests.studies.evidence.properties` and are bounded by the same margins the
implementation comparison declares, so the two halves of the study cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from tests.studies.evidence.document import number
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    PropertyCell,
    rate,
    run_cells,
    se_ratio_interval,
    summarize_cells,
)
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import stream_seed

#: Sized by what the claim needs, not by habit.  The 99% interval's half-width is about
#: ``2.6 / sqrt(m)`` empirical standard deviations, so ``m`` decides the largest bias the study
#: can still place inside the 0.25 margin: 300 replications resolve 0.10, and 1,200 resolve
#: 0.176.  The "only the treatment mechanism correct" cell really does carry about 0.14 of a
#: standard deviation at ``n = 700`` -- it is the arm of double robustness that leans on the
#: inverse-weighting term -- so at 300 the study could not say whether that was inside the
#: margin, and said so.  Raising the budget cannot buy a pass for a bad estimator: the interval
#: contracts on the truth, so a bias past the margin becomes *discriminated* instead.
DOUBLE_ROBUST_REPLICATES = 1_200

#: Sized by what the *coverage* gate in these cells needs, which is the binding one and was
#: the budget's real constraint.  A 99% Clopper-Pearson lower endpoint clearing 0.90 needs
#: 742 of 800 covered replications (92.75%); at 200 it needed 191 (95.5%), which a correctly
#: calibrated 95% interval misses 54.5% of the time.  Raising the budget does not buy a pass:
#: an estimator whose true coverage is 0.90 is refused at 800 with probability 0.996, the same
#: as at 200.  What it buys is that a *passing* cell means something -- the committed n = 500
#: cell sat at exactly 191 of 200, one replication from red for no reason a commit caused.
RATE_REPLICATES = 800
NULL_REPLICATES = 400

#: The calibration cell below.  The 99% resampling interval for the SE ratio has half-width
#: about ``2.58 / sqrt(2 * (m - 1))`` -- 3.7% at 2,400 -- so it leaves about half of the 7%
#: equivalence margin for a real departure rather than spending all of it on Monte Carlo error.
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2000

#: Three sizes, not two: a rate estimated from two points is a ratio with no residual, and
#: quadrupling twice separates a root-n contraction from a merely decreasing one.
RATE_SIZES = (500, 2000, 8000)

#: The expected contraction rate of the sampling distribution, and the alternative the
#: interval must exclude for the check to have discriminated anything.
ROOT_N_SLOPE = -0.5
EXCLUDED_SLOPE = -0.25

#: How far from ``ROOT_N_SLOPE`` the fitted interval may reach and still be accepted.  Derived
#: rather than chosen: half the distance to ``EXCLUDED_SLOPE``, the alternative this study was
#: built to reject, so the accept and the discriminate verdicts partition the line between the
#: two rates the claim is about.  Requiring the interval to *contain* -1/2 instead is the point
#: test the framework refuses everywhere else: it gets harder as replications are added, and
#: the reported-SE rate published at 200 replications per size cleared it by 4.4e-5.
ROOT_N_SLOPE_MARGIN = 0.125

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
                role="control" if name == "both_wrong" else "positive",
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
            property="interval_calibration",
            cell="correctly_specified",
            dgp=linear,
            outcome_learner=LinearRegression,
            treatment_learner=lambda: LogisticRegression(max_iter=1000),
            n=CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES,
            seed=11_100,
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


#: Columns every study family's property summary carries, whatever else it adds.
SHARED_COLUMNS = (
    "rate_sizes",
    "slope",
    "slope_ci_lower",
    "slope_ci_upper",
    "se_ratio_ci_lower",
    "se_ratio_ci_upper",
)


def apply_shared_verdicts(
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    extra_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Every cell the study families share, plus the two rate rows they both publish.

    Each verdict is an interval statement against a margin declared before the run.  For the
    double-robustness cells that means the positive cells must establish that the bias is
    *inside* the margin and the both-wrong control must establish that it is *outside* it --
    the same instrument in both directions, so neither can be passed by a study too small to
    say anything.

    Shared rather than copied per study: the CV reports inherit these cells from
    :func:`cells`, so a verdict written twice is a verdict that can be *changed* once.  The
    caller supplies whatever further columns its own cells publish, because the rate rows are
    built from the summary's columns and have to be built after they all exist.
    """
    margins = record.margins
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
    )
    for column in (*SHARED_COLUMNS, *extra_columns):
        summary[column] = np.nan
    summary["passed"] = False
    # Object dtype, not the NaN float the numeric columns above get: a property with a
    # cross-row claim writes booleans into this, and ``.loc`` will not put one into a
    # float64 column.  ``None`` means "no joint claim beyond this row's own", which
    # :func:`finish` resolves.
    summary["property_passed"] = pd.Series([None] * len(summary), dtype=object, index=summary.index)

    robustness = summary["property"] == "double_robustness"
    positive = robustness & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    control = robustness & (summary["role"] == "control")
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    efficiency = summary["property"] == "root_n_and_efficiency"
    summary.loc[efficiency, "passed"] = (
        (summary.loc[efficiency, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[efficiency, "se_ratio"].between(*margins.se_ratio_sanity)
        & summary.loc[efficiency, "bias_equivalent"]
    )

    calibration = summary["property"] == "interval_calibration"
    for cell in sorted(summary.loc[calibration, "cell"]):
        group = rows.loc[(rows["property"] == "interval_calibration") & (rows["cell"] == cell)]
        ratio = se_ratio_interval(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(record, "interval_calibration", cell),
        )
        mask = calibration & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = ratio.low
        summary.loc[mask, "se_ratio_ci_upper"] = ratio.high
        row = summary.loc[mask].iloc[0]
        coverage = Interval(float(row["coverage_ci_lower"]), float(row["coverage_ci_upper"]))
        # Two-sided, and both halves needed.  A reported standard error inflated by a
        # constant keeps coverage inside its band while failing the ratio; a curve that is
        # right on average but wrong replication by replication does the reverse.
        summary.loc[mask, "passed"] = bool(
            ratio.within(*margins.calibration_se_ratio)
            and coverage.within(*margins.calibration_coverage)
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
            seed=stream_seed(record, "root_n_rate", cell),
        )
        row: dict[str, Any] = dict.fromkeys(summary.columns, np.nan)
        row.update(
            {
                "property": "root_n_rate",
                "cell": cell,
                "role": "positive",
                "n": max(RATE_SIZES),
                "replicates": RATE_REPLICATES * len(RATE_SIZES),
                # The slope is fitted across all three sizes.  ``n`` and ``replicates`` above
                # are the largest and the sum, which read as one big cell; this is what the
                # published table shows instead.
                "rate_sizes": ";".join(f"{size:,}" for size in RATE_SIZES),
                "failed_replicates": 0,
                "slope": fitted.slope,
                "slope_ci_lower": fitted.interval.low,
                "slope_ci_upper": fitted.interval.high,
                "passed": bool(
                    fitted.equivalent_to(ROOT_N_SLOPE, ROOT_N_SLOPE_MARGIN)
                    and fitted.excludes(EXCLUDED_SLOPE)
                ),
            }
        )
        rates.append(row)
    return summary, rates


def decision_rule(record: StudyRecord, row: pd.Series) -> str:
    """The predeclared rule this row's verdict was read against, in the reader's words.

    Authored here rather than in :mod:`tests.studies.evidence.document` because this is where
    the thresholds are declared.  The renderer used to re-type them -- ``"99% slope CI inside
    [-0.6250, -0.3750]"`` and ``"rejection lower >= 0.8000"`` were string literals -- and the
    gate that checks the published page reads the same literal on both sides, so moving
    :data:`ROOT_N_SLOPE_MARGIN` or :data:`MINIMUM_POWER` published a rule the study had not
    applied and nothing failed.  Formatting from the constants makes that impossible instead
    of merely unlikely, and the module's content is hashed into the study manifest.
    """
    margins = record.margins
    property_name = str(row["property"])
    if property_name == "double_robustness":
        # The control's rule is the positive cells' rule reversed: the same instrument has to
        # say "outside" here, or a study too small to resolve anything would pass both.
        direction = "outside" if str(row["role"]) == "control" else "inside"
        return f"99% bias CI {direction} ±{number(row['bias_margin'])}"
    if property_name == "root_n_and_efficiency":
        return (
            f"bias equivalent; coverage lower ≥ {number(margins.coverage_floor)}; "
            f"SE ratio in [{number(margins.se_ratio_sanity[0])}, "
            f"{number(margins.se_ratio_sanity[1])}]"
        )
    if property_name == "root_n_rate":
        return (
            f"99% slope CI inside [{number(ROOT_N_SLOPE - ROOT_N_SLOPE_MARGIN)}, "
            f"{number(ROOT_N_SLOPE + ROOT_N_SLOPE_MARGIN)}] and excluding "
            f"{number(EXCLUDED_SLOPE)}"
        )
    if property_name == "interval_calibration":
        return (
            f"coverage CI in [{number(margins.calibration_coverage[0])}, "
            f"{number(margins.calibration_coverage[1])}]; SE-ratio CI in "
            f"[{number(margins.calibration_se_ratio[0])}, "
            f"{number(margins.calibration_se_ratio[1])}]"
        )
    if property_name == "type_i_error":
        return (
            f"rejection upper ≤ {number(margins.alpha + margins.type_i_margin)}; "
            f"coverage lower ≥ {number(margins.coverage_floor)}"
        )
    if property_name == "power":
        return f"99% rejection lower ≥ {number(MINIMUM_POWER)}"
    raise ValueError(f"no declared decision rule for property {property_name!r}")


def finish(summary: pd.DataFrame, rates: list[dict[str, Any]]) -> pd.DataFrame:
    """Append the rate rows and put the table in its published order.

    ``property_passed`` defaults to the row's own verdict and is overwritten only by a
    property whose claim needs more than one cell to establish -- currently just
    ``crossfit_overfitting``, whose coverage-gain statement is about the *pair*.  Publishing
    both columns is what lets a row state its own rule without losing the joint claim: the
    alternative, broadcasting one scalar across the property, made a deliberately in-sample
    control report the cross-fit arm's verdict as though it were its own.
    """
    summary = pd.concat([summary, pd.DataFrame(rates)], ignore_index=True)
    summary["passed"] = summary["passed"].astype(bool)
    # ``where`` rather than a whole-column default: the rate rows are built from
    # ``dict.fromkeys(summary.columns, nan)``, and a property with a cross-row claim wrote
    # the column only on its own rows.  Everything else is left unset, and ``astype(bool)``
    # would silently publish both ``NaN`` and ``None`` as ``True``.
    joint = summary["property_passed"]
    summary["property_passed"] = joint.where(joint.notna(), summary["passed"]).astype(bool)
    return summary.sort_values(["property", "cell"], ignore_index=True)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-cell descriptive summary, the two rate rows, and every verdict."""
    return finish(*apply_shared_verdicts(rows, STUDY))
