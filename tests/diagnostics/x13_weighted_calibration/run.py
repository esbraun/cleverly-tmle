"""The red weighted calibration cell of ``point-treatment-survival``, refitted on fresh draws.

The registered cell ``interval_calibration/weighted_t5__correctly_specified`` read an SE ratio of
1.027 over 1,600 replications, with a 99% interval of 0.981 to 1.076 against the band's upper end
of 1.07.  This diagnostic asks whether the excess belongs to the fit or to that draw set.  It
runs the cell's own fit (``point_survival_properties.fit_label`` on ``weighted_t5``, in sample,
saturated cell means, the weight ``1 + W1 / 2``) on 1,000 fresh draws of n = 2,000.

Replication ``r`` draws the seed ``9_400_000 + r``, outside every declared stream of the study.
Those are the seeds of the post-run review's probe, so the rows reproduce the numbers the page and
the owner row quote.  The diagnostic cannot change the cell's verdict.

The rows go to ``rows.csv.gz`` and the reading to ``reading.csv``.
``tests/unit/test_point_survival_method_study.py`` rebuilds the reading from the rows.

    python -m tests.diagnostics.x13_weighted_calibration.run --jobs 2
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import point_survival_properties as base

HERE = Path(__file__).resolve().parent
LABEL = "weighted_t5"
N = 2_000
REPLICATES = 1_000
SEED_BASE = 9_400_000


def _fit(replicate: int) -> dict[str, float | int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = base.draw(LABEL, N, SEED_BASE + replicate)
        estimate = base.estimate_of(base.fit_label(LABEL, frame), LABEL)
    return {
        "replicate": replicate,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
    }


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """The SE ratio, both efficiency ratios and the coverage of the fresh draws."""
    estimate = rows["estimate"].to_numpy()
    error = rows["std_error"].to_numpy()
    spread = float(estimate.std(ddof=1))
    bound = base.EFFICIENCY_SD[LABEL]
    truth = base.TRUTHS[LABEL][base.ESTIMAND[LABEL]]
    return pd.DataFrame(
        [
            {
                "replicates": len(rows),
                "se_ratio": float(error.mean()) / spread,
                "empirical_efficiency_ratio": spread * np.sqrt(N) / bound,
                "reported_efficiency_ratio": float(error.mean()) * np.sqrt(N) / bound,
                "coverage": float(np.mean(np.abs(estimate - truth) <= 1.959964 * error)),
                "bias": float(estimate.mean()) - truth,
            }
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    rows = pd.DataFrame(map_parallel(_fit, list(range(REPLICATES)), n_jobs=arguments.jobs))
    rows = rows.sort_values("replicate", ignore_index=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        arguments.output / "rows.csv.gz", index=False, float_format="%.17g", lineterminator="\n"
    )
    reading(rows).to_csv(
        arguments.output / "reading.csv", index=False, float_format="%.6g", lineterminator="\n"
    )


if __name__ == "__main__":
    main()
