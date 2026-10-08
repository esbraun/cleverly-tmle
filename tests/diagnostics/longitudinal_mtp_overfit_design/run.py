"""The pre-run probe of the redesigned overfitting pair of ``longitudinal-mtp``.

Run 2 published the pair red: the in-sample control's SE ratio reached its 0.75 ceiling, and
the paired coverage gain, 0.116 to 0.136, missed its 0.15 floor at its point estimate.  With a
single fully grown tree, the plug-in at a shifted dose is one training outcome, so the in-sample
influence curve still carries the outcome noise.  The third declaration fits an interpolating
extra-trees ensemble in both arms (``longitudinal_mtp_common.OVERFIT_ENSEMBLE``).

This probe fits both arms of the redesigned pair on fresh draws at the cell's size, seeded
apart from every declared stream.  ``tests/unit/test_longitudinal_mtp_design.py`` reads the rows
and checks the expected discrimination before the run.  The probe cannot change a verdict.

    python -m tests.diagnostics.longitudinal_mtp_overfit_design.run --jobs 4
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import longitudinal_mtp_common as common
from tests.studies import longitudinal_mtp_properties as properties
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
REPLICATES = 400


def _seed(replicate: int) -> int:
    return stream_seed(properties.STUDY, "diagnostic", "longitudinal_mtp_overfit_design", replicate)


def _fit(replicate: int) -> dict[str, Any]:
    frame = common.sample_continuous(properties.OVERFIT_N, _seed(replicate), noise=True)
    row: dict[str, Any] = {"replicate": replicate}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for arm, configuration, folds in (
            ("cross_fitted", "overfit_crossfit", 5),
            ("in_sample", "overfit_control", 1),
        ):
            estimate = properties.continuous_fit(frame, configuration, n_folds=folds)[properties.UP]
            row[f"{arm}_estimate"] = float(estimate.psi)
            row[f"{arm}_std_error"] = float(estimate.std_error)
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    rows = map_parallel(_fit, list(range(REPLICATES)), n_jobs=arguments.jobs)
    arguments.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("replicate").to_csv(
        arguments.output / "rows.csv.gz", index=False, float_format="%.17g"
    )


if __name__ == "__main__":
    main()
