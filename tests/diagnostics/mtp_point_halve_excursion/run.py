"""The red calibration cell of ``policy-point-mtp``, refitted on fresh draws.

The registered cell ``interval_calibration/halve__correctly_specified`` read an SE ratio of
1.0429 over 2,000 replications, above its band's upper end of 1.07 at the 99% interval.  This
diagnostic asks whether the excess belongs to the fit or to that one draw set.  It runs the
registered fit (``study.fit`` on the halve pair, both nuisances correct, n = 2,000) on 2,000
fresh draws, seeded apart from every declared stream.  It is a diagnostic only: it cannot change
the cell's verdict, and its replication count equals the cell's.

The rows go to ``rows.csv.gz``.  ``tests/unit/test_policy_point_mtp_design.py`` rebuilds the
reading from them.

    python -m tests.diagnostics.mtp_point_halve_excursion.run --jobs 16
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_policy_point_mtp as study
from tests.studies import policy_point_mtp_properties as properties
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
N = 2_000
REPLICATES = 2_000


def _seed(replicate: int) -> int:
    return stream_seed(study.STUDY, "diagnostic", "mtp_point_halve_excursion", replicate)


def _fit(replicate: int) -> dict[str, float | int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.fit(
            study.sample(N, _seed(replicate)), chosen=properties._pair("halve below 3")
        )
    estimate = result[properties.HALVE]
    return {
        "replicate": replicate,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    rows = map_parallel(_fit, list(range(REPLICATES)), n_jobs=arguments.jobs)
    frame = pd.DataFrame(rows).sort_values("replicate")
    arguments.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output / "rows.csv.gz", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
