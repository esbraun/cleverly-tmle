"""The red ``both_correct__n_2000`` cell of ``longitudinal-sdr``, refitted beside LTMLE.

The registered cell ``sequential_double_robustness/both_correct__n_2000`` read a coverage of 0.911
over 1,000 replications, with a 99% interval of 0.885 to 0.933 against a floor of 0.90.  Its
n = 8,000 partner reads 0.950.  This diagnostic asks whether the shortfall belongs to ``SDR`` or to
the law at n = 2,000.  It runs the cell's own fit (``sdr_family_fit`` with every nuisance correct)
and the shipped ``LTMLE`` with the same learners on the same 1,000 fresh draws of n = 2,000.

Replication ``r`` draws the seed ``100_000 + r`` of :func:`tests.discrete_law_longitudinal_sdr.sample`,
the design-probe range that no declared stream reads.  The diagnostic cannot change the cell's
verdict.

The rows go to ``rows.csv.gz`` and the reading to ``reading.csv``.
``tests/unit/test_sdr_shortfall_reading.py`` rebuilds the reading from the rows.

    python -m tests.diagnostics.x4_both_correct.run --jobs 4
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.longitudinal import LTMLE, SDR
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_sdr as law
from tests.studies import longitudinal_sdr_properties as base

HERE = Path(__file__).resolve().parent
N = 2_000
REPLICATES = 1_000
SEED_BASE = 100_000
ENGINES = {"sdr": SDR, "ltmle": LTMLE}
CRITICAL = 1.959964


def _fit(replicate: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {"replicate": replicate}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = law.sample(N, SEED_BASE + replicate)
        for name, engine in ENGINES.items():
            estimate = base.sdr_family_fit(frame, base.PATTERN_NUISANCES["both_correct"], engine)
            row[f"{name}_estimate"] = float(estimate[base.STATIC].psi)
            row[f"{name}_std_error"] = float(estimate[base.STATIC].std_error)
    return row


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """Coverage, SE ratio and the error-SE correlation of each engine on the fresh draws."""
    truth = float(law.TRUTH[base.STATIC])
    out = []
    for name in ENGINES:
        estimate = rows[f"{name}_estimate"].to_numpy()
        error = rows[f"{name}_std_error"].to_numpy()
        out.append(
            {
                "engine": name,
                "replicates": len(rows),
                "coverage": float(np.mean(np.abs(estimate - truth) <= CRITICAL * error)),
                "se_ratio": float(error.mean()) / float(estimate.std(ddof=1)),
                "se_cv": float(error.std(ddof=1)) / float(error.mean()),
                "error_se_correlation": float(np.corrcoef(error, estimate - truth)[0, 1]),
                "bias": float(estimate.mean()) - truth,
            }
        )
    return pd.DataFrame(out)


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
