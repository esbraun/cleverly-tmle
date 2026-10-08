"""The efficiency red of ``longitudinal-mtp``'s categorical calibration cell, refitted.

``interval_calibration/categorical_mtp__correctly_specified`` read an empirical efficiency
ratio of 1.120 and a reported one of 1.098 at n = 2,000, against a band of 0.9 to 1.1, in run 2
and again in run 3 (the artifacts are identical).  This diagnostic separates three causes:

=================  ===========================================================================
cause              what it predicts
=================  ===========================================================================
a wrong bound      both ratios stay above one as ``n`` grows, in sample and cross-fitted
an inefficient     the same: an influence curve that is not the efficient one keeps its excess
curve
a finite-sample    the ratios fall toward one as ``n`` grows
cost
=================  ===========================================================================

It refits the cell's declared configuration (``properties.categorical_fit``: a logistic
outcome GLM, least squares for the pseudo-outcome and saturated cell probabilities for the
mechanism) on fresh draws, seeded apart from every declared stream.  Each fit runs in sample, as
declared, and cross-fitted at five folds, at n = 2,000 and n = 8,000.  The rows go to
``rows.csv.gz``.  ``tests/unit/test_longitudinal_mtp_reading.py`` rebuilds the reading.  The
diagnostic cannot change the verdict.

    python -m tests.diagnostics.longitudinal_mtp_categorical_efficiency.run --jobs 2
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_longitudinal_mtp as study
from tests.studies import longitudinal_mtp_common as common
from tests.studies import longitudinal_mtp_properties as properties
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
SIZES = (2_000, 8_000)
REPLICATES = 300
FOLDS = {"in_sample": 1, "cross_fitted": 5}


def _seed(n: int, replicate: int) -> int:
    return stream_seed(
        properties.STUDY, "diagnostic", "longitudinal_mtp_categorical_efficiency", n, replicate
    )


def _fit(n: int, replicate: int) -> list[dict[str, Any]]:
    frame = common.CATEGORICAL_LAW.sample(n, _seed(n, replicate))
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for arm, folds in FOLDS.items():
            estimate = study.fit(frame, study.CATEGORICAL, n_folds=folds)[
                properties.CATEGORICAL_NAME
            ]
            rows.append(
                {
                    "arm": arm,
                    "n": n,
                    "replicate": replicate,
                    "estimate": float(estimate.psi),
                    "std_error": float(estimate.std_error),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    payloads = [(n, replicate) for n in SIZES for replicate in range(REPLICATES)]
    rows = [row for part in map_parallel(_fit, payloads, n_jobs=arguments.jobs) for row in part]
    arguments.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["arm", "n", "replicate"]).to_csv(
        arguments.output / "rows.csv.gz", index=False, float_format="%.17g"
    )


if __name__ == "__main__":
    main()
