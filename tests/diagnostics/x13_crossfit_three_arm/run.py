"""The red three-arm calibration cell of ``point-treatment-survival-crossfit``, by size and folds.

The registered cell ``interval_calibration/three_arm_t3__correctly_specified`` read an empirical
efficiency ratio of 1.058, with a 99% interval of 1.010 to 1.108 against the band's upper end of
1.10, and a reported efficiency ratio of 1.067.  This diagnostic asks whether the excess belongs
to the construction or to the sample size.  It refits the cell's own configuration
(``point_survival_properties.fit_label`` on ``three_arm_t3``, saturated cell means) on fresh
draws in five designs, and reads the mean reported standard error over the exact bound.

=================  =====  ============
design             folds  replications
=================  =====  ============
n = 2,000          1      60
n = 2,000          5      60
n = 2,000          10     60
n = 8,000          5      40
n = 8,000          1      20
=================  =====  ============

Replication ``r`` of design ``i`` (in the order above) draws the seed ``9_100_000 + 1_000 i + r``.
Those seeds sit outside every declared stream of the study.  They are the seeds of the pre-page
probe, so the rows reproduce the numbers the page and the owner row quote.  The diagnostic cannot
change the cell's verdict.

The rows go to ``rows.csv.gz`` and the reading to ``reading.csv``.
``tests/unit/test_point_survival_crossfit_method_study.py`` rebuilds the reading from the rows.

    python -m tests.diagnostics.x13_crossfit_three_arm.run --jobs 2
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
LABEL = "three_arm_t3"
#: ``(n, folds, replications)``, in seed order.
DESIGNS: tuple[tuple[int, int, int], ...] = (
    (2_000, 1, 60),
    (2_000, 5, 60),
    (2_000, 10, 60),
    (8_000, 5, 40),
    (8_000, 1, 20),
)
SEED_BASE = 9_100_000


def seed(design: int, replicate: int) -> int:
    """The seed of one replication of one design."""
    return SEED_BASE + 1_000 * design + replicate


def _fit(payload: tuple[int, int, int, int]) -> dict[str, float | int]:
    design, n, folds, replicate = payload
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = base.draw(LABEL, n, seed(design, replicate))
        estimate = base.estimate_of(base.fit_label(LABEL, frame, n_folds=folds), LABEL)
    return {
        "n": n,
        "folds": folds,
        "replicate": replicate,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
    }


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """The mean reported standard error over the exact bound, by design."""
    bound = base.EFFICIENCY_SD[LABEL]
    out = []
    for (n, folds), group in rows.groupby(["n", "folds"], sort=False):
        ratio = group["std_error"].to_numpy() * np.sqrt(n) / bound
        out.append(
            {
                "n": int(n),
                "folds": int(folds),
                "replicates": len(group),
                "reported_over_bound": float(ratio.mean()),
                "reported_over_bound_se": float(ratio.std(ddof=1) / np.sqrt(len(group))),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    payloads = [
        ((design, n, folds, replicate),)
        for design, (n, folds, replications) in enumerate(DESIGNS)
        for replicate in range(replications)
    ]
    rows = pd.DataFrame(map_parallel(_fit, payloads, n_jobs=arguments.jobs))
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        arguments.output / "rows.csv.gz", index=False, float_format="%.17g", lineterminator="\n"
    )
    reading(rows).to_csv(
        arguments.output / "reading.csv", index=False, float_format="%.6g", lineterminator="\n"
    )


if __name__ == "__main__":
    main()
