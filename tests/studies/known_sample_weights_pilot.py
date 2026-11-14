"""The pilot of ``known-mechanism-sample-weights`` (Fix-16 plan section 7, rev 2.1 items 1 and 2).

Before the declaration every gated cell is read on :data:`DRAWS` pilot draws at its declared n,
learners and fit, with the control and the reporting cell on the same draws as the gated point
cell of each size.  The draws come from :data:`PILOT_SEED`, a stream no study draws from, and
each draw is fitted by :mod:`tests.studies.known_sample_weights_law` exactly as the property
phase fits it.

The committed summary :data:`SUMMARY` holds one row per pilot cell: the draws, the coverage of
the shipped interval, the bias against the exact truth, the SD of the estimates, the
standardized bias, the mean reported SE and the SE ratio, the kurtosis of the estimates, the
coefficient of variation of the reported SE, the per-draw spread of the log SE ratio
(:func:`ratio_spread`), the mean Kish effective sample size over n, and the mean seconds per
fit.  The control row also carries the mean paired difference of the control and the reporting
cell on the same draws, over the control's SD.  The design test reads the summary
(``tests/unit/test_known_sample_weights_design.py``).

Run ``python -m tests.studies.known_sample_weights_pilot run --block B --start S --draws D --out
FILE`` once per block of draws, and ``python -m tests.studies.known_sample_weights_pilot summarize
FILE...`` to write the summary.  The per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

from tests.studies import known_sample_weights_law as law
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.registry import Margins
from tests.studies.evidence.schema import reported_inference

#: The pilot draws of every block: ``repeated_sizing.CALIBRATION_PILOT_DRAWS``.
DRAWS = 8_000
#: The pilot's own seed stream.
PILOT_SEED = 2026111431
SUMMARY = Path(__file__).with_name("known_sample_weights_pilot.csv")

#: Each pilot block: ``(law, n, fits)``, the fits made on one draw.  A point block fits the
#: gated cell, the control and the reporting cell on the same draw.
POINT_FITS = ("sample_correct", "target_main", "sample_main")
BLOCKS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    **{f"sw_{n}": ("SW", n, POINT_FITS) for n in (1_000, 4_000, 16_000)},
    "lw_4000": ("LW", 4_000, ("ltmle_sample",)),
}


def pilot_seed(block: str, draw: int) -> int:
    """The seed of one pilot draw of one block."""
    key = sum(ord(character) * 31**index for index, character in enumerate(block)) % 2**31
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(key, draw))
    return int(sequence.generate_state(1)[0])


def kish(weights: np.ndarray) -> float:
    """The Kish effective sample size over n of one draw's weights."""
    return float(weights.sum() ** 2 / np.sum(weights**2) / len(weights))


def records(block: str, draw: int) -> list[dict[str, Any]]:
    """Every pilot row of one draw of one block."""
    name, n, fits = BLOCKS[block]
    seed = pilot_seed(block, draw)
    if name == "SW":
        frame = law.draw(n, seed)
        truth = law.truth()["ate"]
    else:
        frame = law.draw_longitudinal(n, seed)
        truth = law.longitudinal_truth()[law.CONTRAST]
    effective = kish(frame["wt"].to_numpy())
    out = []
    for cell in fits:
        start = time.perf_counter()
        if name == "SW":
            estimate = law.fit(frame, cell)["ate"]
        else:
            estimate = law.fit_longitudinal(frame)[law.CONTRAST]
        seconds = time.perf_counter() - start
        std_error, low, high = reported_inference(estimate)
        out.append(
            {
                "block": block,
                "cell": cell,
                "n": n,
                "draw": draw,
                "truth": truth,
                "estimate": float(estimate.psi),
                "std_error": std_error,
                "covered": int(low <= truth <= high),
                "kish": effective,
                "seconds": seconds,
            }
        )
    return out


def run(block: str, start: int, draws: int, out: Path) -> None:
    """Fit draws ``start`` to ``start + draws - 1`` of one block and write their rows."""
    rows = [row for draw in range(start, start + draws) for row in records(block, draw)]
    pd.DataFrame(rows).to_parquet(out, index=False)


def ratio_spread(estimate: np.ndarray, std_error: np.ndarray) -> float:
    r"""The per-draw SD of the influence of one draw on the log SE ratio.

    The SE ratio is :math:`\bar s / \hat\sigma`.  To first order its log moves by
    :math:`u_i = (s_i - \bar s) / \bar s - ((e_i - \bar e)^2 - \hat\sigma^2) / (2 \hat\sigma^2)`
    per draw, so over ``R`` draws the relative SD of the ratio is :math:`\mathrm{sd}(u) /
    \sqrt R`.  The value carries the kurtosis of the estimates, the spread of the reported SE
    and their covariance.  Normal estimates with an exact SE give :math:`1 / \sqrt 2`.
    """
    centred = estimate - estimate.mean()
    variance = float(np.mean(centred**2))
    u = (std_error - std_error.mean()) / std_error.mean() - (centred**2 - variance) / (
        2.0 * variance
    )
    return float(u.std(ddof=1))


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One summary row per pilot cell."""
    out = []
    for (block, cell), group in rows.groupby(["block", "cell"], sort=True):
        estimate = group["estimate"].to_numpy()
        std_error = group["std_error"].to_numpy()
        sd = float(estimate.std(ddof=1))
        bias = float(estimate.mean() - group["truth"].iloc[0])
        centred = estimate - estimate.mean()
        out.append(
            {
                "block": block,
                "cell": cell,
                "n": int(group["n"].iloc[0]),
                "draws": len(group),
                "truth": float(group["truth"].iloc[0]),
                "bias": bias,
                "sd": sd,
                "bias_over_sd": bias / sd,
                "mean_se": float(std_error.mean()),
                "se_ratio": float(std_error.mean()) / sd,
                "coverage": float(group["covered"].mean()),
                "kurtosis": float(np.mean(centred**4) / np.mean(centred**2) ** 2),
                "se_cv": float(std_error.std(ddof=1) / std_error.mean()),
                "ratio_spread": ratio_spread(estimate, std_error),
                "kish": float(group["kish"].mean()),
                "seconds": float(group["seconds"].mean()),
                "paired_difference_over_sd": np.nan,
            }
        )
    summary = pd.DataFrame(out)
    for block in summary["block"].unique():
        cells = rows.loc[rows["block"] == block]
        if "target_main" not in set(cells["cell"]):
            continue
        wide = cells.pivot(index="draw", columns="cell", values="estimate")
        difference = wide["target_main"] - wide["sample_main"]
        mask = (summary["block"] == block) & (summary["cell"] == "target_main")
        sd = float(summary.loc[mask, "sd"].iloc[0])
        summary.loc[mask, "paired_difference_over_sd"] = float(difference.mean()) / sd
    return summary


# ---------------------------------------------------------------------------- sizing
#: Gauss-Hermite nodes and weights (probabilists') of the predictive integrals.
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(31)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def se_half_pass(margins: Margins, replicates: int, ratio: float, spread: float) -> float:
    """P(the 99% interval of the SE ratio lies inside the calibration band).

    The estimated ratio is normal with relative SD ``spread / sqrt(R)`` (:func:`ratio_spread`
    measured on the pilot, which carries the kurtosis of the estimates and the spread of the
    reported SE), and the bootstrap interval has half-width ``z_{0.995}`` of that.
    :func:`tests.studies.repeated_sizing.calibration_pass` is the special case
    ``spread = 1 / sqrt(2)`` with an exact SE.
    """
    low, high = margins.calibration_se_ratio
    z = float(norm.ppf(0.5 + margins.confidence_level / 2.0))
    scale = ratio * spread / np.sqrt(replicates)
    half = z * scale
    return max(
        float(norm.cdf((high - half - ratio) / scale) - norm.cdf((low + half - ratio) / scale)),
        0.0,
    )


def coverage_half_pass(margins: Margins, replicates: int, coverage: float) -> float:
    """P(the exact 99% interval of the coverage lies inside the calibration coverage band)."""
    accepted = sizing.accepted_counts(margins, replicates)
    if accepted is None:
        return 0.0
    return float(
        binom.cdf(accepted[1], replicates, coverage)
        - binom.cdf(accepted[0] - 1, replicates, coverage)
    )


def calibration_cell_pass(
    margins: Margins, replicates: int, ratio: float, coverage: float, spread: float
) -> float:
    """The calibration positive's pass probability: the SE half times the coverage half."""
    return se_half_pass(margins, replicates, ratio, spread) * coverage_half_pass(
        margins, replicates, coverage
    )


def predictive_calibration_pass(margins: Margins, replicates: int, row: pd.Series) -> float:
    """:func:`calibration_cell_pass` averaged over the pilot noise of the ratio and coverage.

    The true ratio is normal about the pilot's with relative SD ``ratio_spread / sqrt(D)`` and
    the true coverage normal about the pilot's with its binomial SD, independently.
    """
    draws = int(row["draws"])
    ratio, coverage, spread = (
        float(row["se_ratio"]),
        float(row["coverage"]),
        float(row["ratio_spread"]),
    )
    ratios = ratio * (1.0 + spread / np.sqrt(draws) * _NODES)
    coverages = np.clip(coverage + np.sqrt(coverage * (1.0 - coverage) / draws) * _NODES, 0.0, 1.0)
    se = np.array([se_half_pass(margins, replicates, r, spread) for r in ratios])
    cover = np.array([coverage_half_pass(margins, replicates, c) for c in coverages])
    return float(np.sum(_WEIGHTS * se) * np.sum(_WEIGHTS * cover))


def predictive_bias_pass(margins: Margins, replicates: int, row: pd.Series) -> float:
    """The bias positive's pass probability averaged over the pilot noise of its bias."""
    biases = float(row["bias_over_sd"]) + _NODES / np.sqrt(int(row["draws"]))
    return float(np.sum(_WEIGHTS * sizing.bias_equivalent_probability(margins, biases, replicates)))


def predictive_control_pass(margins: Margins, replicates: int, row: pd.Series) -> float:
    """The control's pass probability averaged over the pilot noise of its bias."""
    gaps = float(row["bias_over_sd"]) + _NODES / np.sqrt(int(row["draws"]))
    values = [sizing.bias_discriminated_probability(margins, gap, replicates) for gap in gaps]
    return float(np.sum(_WEIGHTS * np.asarray(values)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--block", choices=sorted(BLOCKS), required=True)
    runner.add_argument("--start", type=int, required=True)
    runner.add_argument("--draws", type=int, required=True)
    runner.add_argument("--out", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("files", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        run(arguments.block, arguments.start, arguments.draws, arguments.out)
        return
    rows = pd.concat([pd.read_parquet(path) for path in arguments.files], ignore_index=True)
    rows = rows.drop_duplicates(["block", "cell", "draw"])
    summarize(rows).to_csv(SUMMARY, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
