"""Run the declared RM18 four-cell density-ratio replay outside registered artifacts."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.canonical.regenerate import Reference
from tests.studies.canonical_shift_policies import (
    POLICIES,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    SCENARIO,
    draw_scenario,
    fit_cleverly,
    initial_estimates,
)
from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.registry import ROOT
from tests.studies.point_study_helpers import primary_rows

TARGET = "ate_shift[+0.25 vs natural course]"
CELLS = ("C-H", "C-A", "R-A", "R-H")
ANCHORED = {"C-H": "cleverly", "R-A": "lmtp"}
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20260926
EQUIVALENCE = 0.01
ANCHOR_TOLERANCE = 1e-6
CONTRASTS: dict[str, dict[str, int]] = {
    "observed_gap": {"C-H": 1, "R-A": -1},
    "C_density": {"C-H": 1, "C-A": -1},
    "R_density": {"R-H": 1, "R-A": -1},
    "analytic_engine": {"C-A": 1, "R-A": -1},
    "hazard_engine": {"C-H": 1, "R-H": -1},
    "interaction": {"C-H": 1, "C-A": -1, "R-H": -1, "R-A": 1},
}
ANCHOR = ROOT / "tests" / "canonical" / "lmtp_shift" / "replicates.csv.gz"
REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="tests/diagnostics/rm18_comparator_density/run.R",
    mount_runner=True,
    extra_files=("tests/canonical/lmtp_point_adapter.R", "tests/canonical/study_harness.R"),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT,
)


@dataclass(frozen=True)
class Interval:
    """A closed interval used for simultaneous interval arithmetic."""

    lower: float
    upper: float


def _policy_ratio(
    dose: np.ndarray, mean: np.ndarray, delta: float, cap: float | None
) -> np.ndarray:
    """The analytic normal-density ratio, including both capped-policy preimages."""
    ratio = np.exp(delta * (dose - mean) - 0.5 * delta**2)
    if cap is not None:
        ratio = ratio * (dose <= cap) + (dose > cap - delta)
    if not np.all(np.isfinite(ratio)) or np.any(ratio < 0):
        raise ValueError("the analytic shift ratio is non-finite or negative")
    return np.asarray(ratio, dtype=float)


def analytic_arrays(result: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build observed and shifted-dose ratios in the stored ShiftSet's axis order."""
    shifts = result.nuisance.shifts
    if shifts is None or shifts.names != tuple(name for _, _, name in POLICIES):
        raise ValueError("the fit does not carry the declared three policy axes")
    dose = np.asarray(frame["A"], dtype=float)
    if shifts.shifted.shape != (len(dose), len(POLICIES)):
        raise ValueError("the stored shifted doses do not align with the sample")
    mean = 2.0 + 0.7 * np.asarray(frame["W1"]) - 0.3 * np.asarray(frame["W2"])
    observed = np.column_stack(
        [_policy_ratio(dose, mean, delta, cap) for delta, cap, _ in POLICIES]
    )
    at_shifted = np.stack(
        [
            np.column_stack(
                [
                    _policy_ratio(shifts.shifted[:, s], mean, delta, cap)
                    for delta, cap, _ in POLICIES
                ]
            )
            for s in range(len(POLICIES))
        ],
        axis=1,
    )
    return observed, at_shifted


def hazard_quarter(result: Any, n: int) -> np.ndarray:
    """Take the untruncated observed-dose quarter ratio, with an axis-order guard."""
    shifts = result.nuisance.shifts
    if shifts is None or shifts.names != tuple(name for _, _, name in POLICIES):
        raise ValueError("the fit does not carry the declared three policy axes")
    if shifts.ratio.shape != (n, len(POLICIES)):
        raise ValueError("the stored ratio does not align with the sample")
    quarter = np.asarray(shifts.ratio[:, 1], dtype=float)
    if not np.all(np.isfinite(quarter)) or np.any(quarter < 0):
        raise ValueError("the observed-dose quarter ratio is non-finite or negative")
    return quarter


def analytic_retarget(result: Any, frame: pd.DataFrame) -> Any:
    """Change only the stored density ratios, then rerun the joint targeting step."""
    observed, at_shifted = analytic_arrays(result, frame)
    shifts = replace(result.nuisance.shifts, ratio=observed, ratio_at=at_shifted)
    nuisance = replace(result.nuisance, shifts=shifts)
    estimates, fluctuations = result.estimator.retarget(
        result.data, nuisance, estimands=("ey_shift", "ate_shift")
    )
    repeat = replace(
        result.repeats[0],
        nuisance=nuisance,
        fluctuations=fluctuations,
        psi={name: estimate.psi for name, estimate in estimates.items()},
    )
    return replace(result, estimates=estimates, repeats=(repeat,))


def _python_cell(result: Any, truth: dict[str, float], replicate: int, cell: str) -> dict[str, Any]:
    row = primary_rows(
        result=result,
        truth=truth,
        implementation=cell,
        scenario=SCENARIO,
        replicate=replicate,
        estimands=(TARGET,),
        initials=initial_estimates(result),
        n=PRIMARY_N,
    )[0]
    row["cell"] = row.pop("implementation")
    return row


def _prepare_replicate(replicate: int) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    frame, truth = draw_scenario(SCENARIO, PRIMARY_N, replicate)
    if len(frame) != PRIMARY_N or TARGET not in truth:
        raise RuntimeError(f"replicate {replicate} does not match the registered draw")
    fitted = fit_cleverly(frame)
    rows = [
        _python_cell(fitted, truth, replicate, "C-H"),
        _python_cell(analytic_retarget(fitted, frame), truth, replicate, "C-A"),
    ]
    sample = frame.copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", SCENARIO)
    sample["hazard_ratio_quarter"] = hazard_quarter(fitted, len(frame))
    truth_row = {
        "scenario": SCENARIO,
        "replicate": replicate,
        "estimand": TARGET,
        "truth": truth[TARGET],
    }
    return sample, truth_row, rows


def prepare(samples: Path, truths: Path, python_rows: Path, *, jobs: int, replicates: int) -> None:
    """Redraw the registered seeds, fit C-H, retarget C-A, and supply R-H's ratio."""
    results = map_parallel(_prepare_replicate, range(replicates), n_jobs=jobs)
    frames = [sample for sample, _, _ in results]
    truth_rows = [truth for _, truth, _ in results]
    rows = [row for _, _, pair in results for row in pair]
    write_csv(pd.concat(frames, ignore_index=True), samples, compression="gzip")
    write_csv(pd.DataFrame(truth_rows), truths)
    write_csv(pd.DataFrame(rows), python_rows, compression="gzip")


def _validate_anchor(cells: pd.DataFrame, anchors: pd.DataFrame, cell: str, count: int) -> None:
    implementation = ANCHORED[cell]
    committed = anchors.loc[
        (anchors["implementation"] == implementation)
        & (anchors["scenario"] == SCENARIO)
        & (anchors["estimand"] == TARGET)
        & (anchors["replicate"] < count)
    ]
    if len(committed) != count or committed["replicate"].nunique() != count:
        raise ValueError(f"the committed {cell} anchor has missing or duplicate keys")
    current = cells.loc[cells["cell"] == cell].set_index("replicate").sort_index()
    old = committed.set_index("replicate").sort_index()
    if not current.index.equals(old.index):
        raise ValueError(f"the {cell} anchor uses different replicate keys")
    for column in ("estimate", "std_error", "initial_estimate"):
        if np.max(np.abs(current[column].to_numpy() - old[column].to_numpy())) > ANCHOR_TOLERANCE:
            raise ValueError(f"the {cell} anchor moved in {column}")
    for column in ("n", "covered"):
        if not np.array_equal(current[column].to_numpy(), old[column].to_numpy()):
            raise ValueError(f"the {cell} anchor moved in {column}")
    if not np.array_equal(current["truth"].to_numpy(), old["truth"].to_numpy()):
        raise ValueError(f"the {cell} anchor moved in truth")


def validate_cells(
    cells: pd.DataFrame, anchors: pd.DataFrame, count: int = PRIMARY_REPLICATES
) -> None:
    """Require complete pairs and per-replication agreement with both registered anchors."""
    expected = {(cell, replicate) for cell in CELLS for replicate in range(count)}
    keys = list(zip(cells["cell"], cells["replicate"], strict=True))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("the four-cell replay has missing, duplicate, or unexpected keys")
    numeric = (
        "truth",
        "estimate",
        "inference_estimate",
        "std_error",
        "ci_lower",
        "ci_upper",
        "initial_estimate",
    )
    if not np.all(np.isfinite(cells[list(numeric)].to_numpy(dtype=float))):
        raise ValueError("a replay cell has a non-finite estimate or interval")
    if np.any(cells["std_error"].to_numpy(dtype=float) <= 0):
        raise ValueError("a replay cell has a nonpositive standard error")
    if not np.all(cells["n"].to_numpy() == PRIMARY_N):
        raise ValueError("a replay cell has the wrong sample size")
    if not np.all(cells["scenario"] == SCENARIO) or not np.all(cells["estimand"] == TARGET):
        raise ValueError("a replay cell names another scenario or estimand")
    if not np.all(cells["inference_scale"] == "identity"):
        raise ValueError("a replay cell uses another inference scale")
    covered = (cells["ci_lower"] <= cells["truth"]) & (cells["truth"] <= cells["ci_upper"])
    if not np.array_equal(covered.to_numpy(dtype=int), cells["covered"].to_numpy(dtype=int)):
        raise ValueError("a replay cell has an incorrect coverage flag")
    truths = cells.pivot(index="replicate", columns="cell", values="truth")
    truth_values = truths.to_numpy(dtype=float)
    if not np.all(truth_values == truth_values[:, :1]):
        raise ValueError("four cells of one replicate disagree on truth")

    for cell in ANCHORED:
        _validate_anchor(cells, anchors, cell, count)


def _signed_errors(estimates: np.ndarray, standard_errors: np.ndarray) -> np.ndarray:
    spread = np.std(estimates, axis=-1, ddof=1)
    if not np.all(np.isfinite(spread)) or np.any(spread <= 0):
        raise ValueError("a cell's empirical spread is zero or non-finite")
    return np.mean(standard_errors, axis=-1) / spread - 1.0


def _absolute(interval: Interval) -> Interval:
    if interval.lower >= 0:
        return interval
    if interval.upper <= 0:
        return Interval(-interval.upper, -interval.lower)
    return Interval(0.0, max(-interval.lower, interval.upper))


def _contrast(intervals: dict[str, Interval], coefficients: dict[str, int]) -> Interval:
    lower = sum(
        intervals[cell].lower if sign > 0 else -intervals[cell].upper
        for cell, sign in coefficients.items()
    )
    upper = sum(
        intervals[cell].upper if sign > 0 else -intervals[cell].lower
        for cell, sign in coefficients.items()
    )
    return Interval(lower, upper)


def _equivalent(interval: Interval) -> bool:
    return interval.lower > -EQUIVALENCE and interval.upper < EQUIVALENCE


def _residual(interval: Interval) -> bool:
    return interval.lower > EQUIVALENCE or interval.upper < -EQUIVALENCE


def classify(intervals: dict[str, Interval]) -> str:
    """Apply the declaration's additive-path and equivalence reading rule."""
    if intervals["observed_gap"].lower <= 0:
        return "unresolved"
    density = intervals["C_density"].lower > 0 or intervals["R_density"].lower > 0
    residual = _residual(intervals["analytic_engine"]) or _residual(intervals["hazard_engine"])
    if density and residual:
        return "mixed"
    path_c = intervals["C_density"].lower > 0 and _equivalent(intervals["analytic_engine"])
    path_r = intervals["R_density"].lower > 0 and _equivalent(intervals["hazard_engine"])
    if (path_c or path_r) and _equivalent(intervals["interaction"]):
        return "density sufficient"
    if residual:
        return "engine residual"
    return "unresolved"


def reading(cells: pd.DataFrame, bootstraps: int = BOOTSTRAPS) -> pd.DataFrame:
    """Recompute the four signed intervals, mapped contrasts, and declared label."""
    ordered = cells.sort_values(["cell", "replicate"])
    estimates = np.vstack(
        [
            ordered.loc[ordered["cell"] == cell, "inference_estimate"].to_numpy(dtype=float)
            for cell in CELLS
        ]
    )
    standard_errors = np.vstack(
        [ordered.loc[ordered["cell"] == cell, "std_error"].to_numpy(dtype=float) for cell in CELLS]
    )
    n = estimates.shape[1]
    point = _signed_errors(estimates, standard_errors)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((bootstraps, len(CELLS)), dtype=float)
    for start in range(0, bootstraps, 100):
        stop = min(start + 100, bootstraps)
        indices = rng.integers(0, n, size=(stop - start, n))
        sampled_estimates = estimates[:, indices]
        sampled_standard_errors = standard_errors[:, indices]
        draws[start:stop] = _signed_errors(sampled_estimates, sampled_standard_errors).T
    bounds = np.quantile(draws, (0.00125, 0.99875), axis=0)
    signed = {
        cell: Interval(float(bounds[0, i]), float(bounds[1, i])) for i, cell in enumerate(CELLS)
    }
    absolute = {cell: _absolute(interval) for cell, interval in signed.items()}
    contrasts = {
        name: _contrast(absolute, coefficients) for name, coefficients in CONTRASTS.items()
    }
    label = classify(contrasts)
    point_absolute = {cell: abs(float(value)) for cell, value in zip(CELLS, point, strict=True)}
    rows = [
        {
            "statistic": f"signed_{cell}",
            "estimate": float(point[i]),
            "lower": signed[cell].lower,
            "upper": signed[cell].upper,
            "reading": label,
        }
        for i, cell in enumerate(CELLS)
    ]
    rows.extend(
        {
            "statistic": name,
            "estimate": sum(point_absolute[cell] * sign for cell, sign in coefficients.items()),
            "lower": contrasts[name].lower,
            "upper": contrasts[name].upper,
            "reading": label,
        }
        for name, coefficients in CONTRASTS.items()
    )
    return pd.DataFrame(rows)


def run(output: Path, jobs: int, *, smoke: bool = False) -> None:
    """Run the complete declared replay after its design commit is pushed."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    registered = ANCHOR.parent.resolve()
    resolved = output.resolve()
    if resolved == registered or registered in resolved.parents:
        raise ValueError("the diagnostic output cannot enter registered artifacts")
    count = 1 if smoke else PRIMARY_REPLICATES
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=output) as scratch_name:
        scratch = Path(scratch_name)
        samples = scratch / "samples.csv.gz"
        truths = scratch / "truth.csv"
        python_rows = scratch / "python-rows.csv.gz"
        reference_rows = scratch / "reference-rows.csv"
        prepare(samples, truths, python_rows, jobs=jobs, replicates=count)
        anchors = pd.read_csv(ANCHOR)
        _validate_anchor(pd.read_csv(python_rows), anchors, "C-H", count)
        REFERENCE.run(
            ROOT / "tests" / "diagnostics" / "rm18_comparator_density",
            samples,
            truths,
            reference_rows,
            cores=jobs,
        )
        cells = pd.concat(
            [pd.read_csv(python_rows), pd.read_csv(reference_rows)], ignore_index=True
        )
        validate_cells(cells, anchors, count=count)
        if smoke:
            (output / "smoke.txt").write_text(
                "Both registered anchors and four-cell keys passed.\n"
            )
            return
        summary = reading(cells)
        write_csv(
            cells.sort_values(["cell", "replicate"]), output / "cells.csv.gz", compression="gzip"
        )
        write_csv(summary, output / "reading.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--smoke", action="store_true", help="check one anchored pair; write no reading"
    )
    args = parser.parse_args()
    run(args.output, args.jobs, smoke=args.smoke)


if __name__ == "__main__":
    main()
