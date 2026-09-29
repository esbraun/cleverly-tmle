"""Design CD: attribute the calibration excess of the continuous modified treatment policy row.

RM18 of ``docs/roadmap.md`` declares the design in "Design CD".  It refits the 800 registered
primary draws of ``shift-policies`` with the registered ``cleverly`` fit (arm Cb), targets each
fit again with the analytic density ratio (arm Ca), and reads both beside the committed ``lmtp``
rows (arm L).  It draws no new sample and runs no R.

    python -m tests.diagnostics.rm18_comparator_density.run --part CD --output <scratch>

``--read-only`` rebuilds ``cd-reading.csv`` from ``cd-rows.csv.gz`` and ``cd-validation.csv``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.exceptions import PositivityWarning
from cleverly.interventions.shift import ShiftSet
from tests.diagnostics import rm18_shared as shared
from tests.studies import canonical_shift_policies as study
from tests.studies.evidence.comparison import _bounds
from tests.studies.evidence.inference import Interval, bootstrap
from tests.studies.evidence.seeds import stream_seed
from tests.studies.point_study_helpers import primary_rows

HERE = Path(__file__).resolve().parent
DESIGN = "comparator-density"
PART = "CD"
SHIFT = study.STUDY
ESTIMAND = "ate_shift[+0.25 vs natural course]"
DELTA = 0.25
N = study.PRIMARY_N
DGP = study.shift_dgp(curvature=study.PRIMARY_CURVATURE)
#: ``Var(A)`` on the law: the dose noise plus each standard-normal covariate's squared
#: loading in ``mu(W)``, read off the law's own ``dose_mean``.
_LOADINGS = DGP.dose_mean(np.eye(DGP.n_latent)) - DGP.dose_mean(np.zeros((1, DGP.n_latent)))
DOSE_VARIANCE = float(DGP.dose_scale**2 + np.sum(np.square(_LOADINGS)))
#: The exact efficiency bound of the contrast, from the closed form the declaration gives.
SIGMA = math.sqrt(
    DGP.noise_scale**2 * (math.exp(DELTA**2) - 1.0)
    + (2.0 * study.PRIMARY_CURVATURE * DELTA) ** 2 * DOSE_VARIANCE
)
MARGIN = SHIFT.margins.calibration_noninferiority

BINNED = "cleverly"
ANALYTIC = "cleverly-analytic-density"
COMPARATOR = "lmtp"
ARMS = {"Cb": BINNED, "Ca": ANALYTIC, "L": COMPARATOR}
DIFFERENCES = (("Cb", "L"), ("Ca", "L"), ("Cb", "Ca"))

DENSITY = "cleverly density representation"
TARGETING = "cleverly, not the density"
COMPARATOR_READING = "comparator"
UNRESOLVED = "unresolved"


class AnalyticDensity:
    """The law's conditional dose density, ``phi(a - mu(W))`` with SD 1, on each row.

    ``ShiftSet.evaluate`` reads ``density_at`` for every clever covariate.  It reads
    ``crossing_fraction`` only to warn when a binned density cannot resolve a shift, and the
    exact density resolves every shift, so it reports one.
    """

    def __init__(self, covariates: np.ndarray) -> None:
        self.mean = np.asarray(DGP.dose_mean(covariates), dtype=float)

    def density_at(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(
            norm.pdf(np.asarray(values, dtype=float) - self.mean, scale=DGP.dose_scale)
        )

    def crossing_fraction(self, shifted: np.ndarray, observed: np.ndarray) -> float:
        del shifted, observed
        return 1.0


def analytic_retarget(result: Any, frame: pd.DataFrame) -> Any:
    """The registered fit, targeted again with the analytic density ratio.

    The seam ``canonical_shift_policies.reversed_ratio_control`` uses: replace
    ``nuisance.shifts`` and call ``result.estimator.retarget``.
    """
    covariates = frame[list(DGP.covariate_names)].to_numpy(dtype=float)
    with warnings.catch_warnings():
        # The uncapped-support warning the registered fit already contains, for the same shift.
        warnings.simplefilter("ignore", PositivityWarning)
        # ``evaluate`` reads ``density_at`` and ``crossing_fraction`` alone, which the analytic
        # density supplies; it is not a fitted ``ConditionalDensity``.
        evaluated = ShiftSet.evaluate(
            study.shifts(),
            result.data,
            AnalyticDensity(covariates),  # type: ignore[arg-type]
        )
    shifts = replace(evaluated, reference=result.nuisance.shifts.reference)
    nuisance = replace(result.nuisance, shifts=shifts)
    estimates, _ = result.estimator.retarget(
        result.data, nuisance, estimands=("ey_shift", "ate_shift")
    )
    return replace(result, estimates=estimates)


def refit(replicate: int) -> list[dict[str, Any]]:
    """Arm Cb's rows of every estimand and arm Ca's row of the read estimand, for one draw."""
    frame, truth = study.draw_scenario(study.SCENARIO, N, replicate)
    result = study.fit_cleverly(frame)
    binned = study.cleverly_rows(frame, truth, study.SCENARIO, replicate)
    analytic = primary_rows(
        result=analytic_retarget(result, frame),
        truth=truth,
        implementation=ANALYTIC,
        scenario=study.SCENARIO,
        replicate=replicate,
        estimands=(ESTIMAND,),
        n=len(frame),
    )
    return binned + analytic


def committed_rows() -> pd.DataFrame:
    return shared.read_rows(SHIFT.artifact("replicates.csv.gz"))


def excess(rows: pd.DataFrame, subject: str, reference: str) -> dict[str, float]:
    """The framework calibration excess bound and resolution of one pair, on the registered
    stream ``stream_seed(SHIFT, "equivalence", scenario, estimand)``."""
    pair = rows.loc[rows["implementation"].isin((ARMS[subject], ARMS[reference]))]
    record = replace(SHIFT, implementation=ARMS[subject], reference=ARMS[reference])
    bounds = _bounds(
        (
            record,
            pair,
            ESTIMAND,
            float(pair["truth"].iloc[0]),
            stream_seed(SHIFT, "equivalence", study.SCENARIO, ESTIMAND),
        )
    )
    return {
        "upper": float(bounds["calibration_excess_upper"]),
        "resolution": float(bounds["calibration_excess_resolution"]),
    }


def registered_excess() -> tuple[float, float]:
    """The committed (Cb, L) calibration excess bound and resolution, from ``equivalence.csv``."""
    committed = shared.read_rows(SHIFT.artifact("equivalence.csv")).set_index("estimand")
    row = committed.loc[ESTIMAND]
    return float(row["calibration_excess_upper"]), float(row["calibration_excess_resolution"])


def validate(fitted: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """R4: Cb reproduces every committed ``cleverly`` row, and (Cb, L) its committed excess."""
    committed = committed_rows()
    replicates = set(fitted["replicate"])
    selection = committed.loc[
        (committed["implementation"] == BINNED) & committed["replicate"].isin(replicates)
    ]
    keys = ["replicate", "estimand"]
    rows = shared.compare_rows(fitted.loc[fitted["implementation"] == BINNED], selection, keys)
    out = [shared.validation_row(PART, "Cb refit rows", rows)]
    if cap is None:
        pair = paired_rows(fitted)
        bound = excess(pair, "Cb", "L")
        largest = float(
            np.max(
                shared.scaled_difference(
                    [bound["upper"], bound["resolution"]], list(registered_excess())
                )
            )
        )
        out.append(
            shared.validation_row(PART, "(Cb, L) excess", (largest <= shared.TOLERANCE, largest, 2))
        )
    return pd.DataFrame(out, columns=list(shared.VALIDATION_COLUMNS))


def paired_rows(fitted: pd.DataFrame) -> pd.DataFrame:
    """The three arms' rows of the read estimand, on the replicates the run fitted."""
    committed = committed_rows()
    comparator = committed.loc[
        (committed["implementation"] == COMPARATOR)
        & (committed["estimand"] == ESTIMAND)
        & committed["replicate"].isin(set(fitted["replicate"]))
    ]
    own = fitted.loc[fitted["estimand"] == ESTIMAND]
    return pd.concat([own, comparator], ignore_index=True).loc[
        :,
        [
            "implementation",
            "replicate",
            "truth",
            "estimate",
            "inference_estimate",
            "std_error",
            "covered",
        ],
    ]


def arm_statistics(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    scale = math.sqrt(N) / SIGMA
    out = {}
    for arm, implementation in ARMS.items():
        group = rows.loc[rows["implementation"] == implementation]
        reported = float(group["std_error"].mean()) * scale
        out[arm] = {
            "rho_rep": reported,
            "rho_emp": float(group["estimate"].std(ddof=1)) * scale,
            "D": abs(reported - 1.0),
        }
    return out


def d_intervals(rows: pd.DataFrame) -> dict[tuple[str, str], Interval]:
    """The paired bootstrap of each declared difference in ``D``, from one index matrix."""
    wide = rows.pivot(index="replicate", columns="implementation", values="std_error")
    arrays = {
        arm: wide[implementation].to_numpy(dtype=float) for arm, implementation in ARMS.items()
    }
    scale = math.sqrt(N) / SIGMA

    def distance(arm: str) -> Any:
        return lambda draw: np.abs(draw[arm].mean(axis=1) * scale - 1.0)

    statistics = {
        f"{left} - {right}": (
            lambda draw, left=left, right=right: distance(left)(draw) - distance(right)(draw)
        )
        for left, right in DIFFERENCES
    }
    samples = bootstrap(
        arrays,
        statistics,
        replicates=SHIFT.margins.bootstrap_replicates,
        seed=shared.bootstrap_seed(SHIFT, DESIGN, "D"),
    )
    return {
        (left, right): shared.interval_of(samples[f"{left} - {right}"])
        for left, right in DIFFERENCES
    }


def reading_label(binned: Interval, analytic: Interval, analytic_excess_upper: float) -> str:
    """The declared reading, from ``D_Cb - D_L``, ``D_Ca - D_L`` and the (Ca, L) bound."""
    if binned.low > 0.0 and not analytic.low > 0.0 and analytic_excess_upper <= MARGIN:
        return DENSITY
    if binned.low > 0.0 and analytic.low > 0.0:
        return TARGETING
    if binned.high < 0.0:
        return COMPARATOR_READING
    return UNRESOLVED


def reading_table(rows: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    out = shared.validation_readings(validation, PART)
    if not shared.validated(validation, PART):
        out.append(shared.reading(PART, ESTIMAND, "reading", result=shared.NOT_VALIDATED))
        return shared.reading_frame(out)
    smoke = bool((rows.groupby("implementation").size() != study.PRIMARY_REPLICATES).any())
    out.append(shared.reading(PART, "exact bound", "sigma*", value=SIGMA))
    points = arm_statistics(rows)
    for arm, statistics in points.items():
        for name, value in statistics.items():
            out.append(shared.reading(PART, arm, name, value=value))
    intervals = d_intervals(rows)
    for (left, right), interval in intervals.items():
        out.append(
            shared.reading(
                PART,
                f"D_{left} - D_{right}",
                "difference",
                value=points[left]["D"] - points[right]["D"],
                interval=interval,
            )
        )
    bounds = {pair: excess(rows, *pair) for pair in DIFFERENCES}
    for (left, right), bound in bounds.items():
        out.append(
            shared.reading(
                PART, f"({left}, {right})", "calibration excess bound", value=bound["upper"]
            )
        )
        out.append(
            shared.reading(
                PART,
                f"({left}, {right})",
                "calibration excess resolution",
                value=bound["resolution"],
            )
        )
    label = reading_label(intervals["Cb", "L"], intervals["Ca", "L"], bounds["Ca", "L"]["upper"])
    out.append(shared.reading(PART, ESTIMAND, "reading", result=shared.label(label, smoke)))
    return shared.reading_frame(out)


def run_part(part: str, output: Path, cap: int | None, jobs: int) -> None:
    """Refit the registered draws (Cb and Ca), validate Cb, and write the three arms' rows."""
    rows_path, validation_path, _ = shared.part_paths(output, part)
    replicates = shared.budget(study.PRIMARY_REPLICATES, cap)
    fitted = pd.DataFrame(
        [row for rows in shared.pool(refit, range(replicates), jobs) for row in rows]
    )
    validation = validate(shared.require_finite(fitted), cap)
    shared.write_table(validation, validation_path)
    if shared.validated(validation, part):
        shared.write_table(paired_rows(fitted), rows_path)


def table(part: str, output: Path, jobs: int) -> pd.DataFrame:
    del jobs
    rows_path, validation_path, _ = shared.part_paths(output, part)
    return reading_table(shared.optional_rows(rows_path), shared.read_rows(validation_path))


if __name__ == "__main__":
    shared.main(__doc__.splitlines()[0], (PART,), HERE, run_part, table)
