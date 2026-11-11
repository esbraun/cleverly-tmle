r"""The ``both_correct__n_2000`` cell of ``longitudinal-sdr``, below its tail size, beside LTMLE.

The registered cell ``sequential_double_robustness/both_correct__n_2000`` read a coverage of 0.911
over 1,000 replications, with a 99% interval of 0.885 to 0.933 against a floor of 0.90.  Its
n = 8,000 partner reads 0.950.  On the family law the exact-law tail size of the contrast
``ate_regimen[always vs never]`` is :math:`n_\kappa` = 2,326, so the n of 2,000 sits below even
:math:`n_\kappa`, and the cell's Wald coverage is a reporting reading under the tail-size rule of
the study (``tests/studies/longitudinal_sdr_properties.py``, ``BELOW_TAIL_SIZE``).  The gated
coverage claim for the contrast is the cell at n = 4,800.

The diagnostic runs the cell's own fit (``sdr_family_fit`` with every nuisance correct) and the
shipped ``LTMLE`` with the same learners on the same 1,000 fresh draws of n = 2,000.  Both cover
alike, so the shortfall belongs to the law at this n, not to ``SDR``.

The mechanism is the always-path cell ``W = 1, A1 = 1, C1 = 1, L2 = 1, A2 = 1, C2 = 1``: its
probability is 0.001465 (2.9 rows at n = 2,000), its inverse weight 85.3, and it carries 27% of
the EIF variance.  :func:`strata` splits the draws by what that cell holds: no row, one row, two
or more rows with every outcome 1 or every outcome 0, or mixed outcomes.  In all but the last,
the cell carries no within-cell variation: a cell of equal outcomes is fitted exactly, and an
empty or one-row cell is absent from the training folds of its held-out row, so the cell mean
falls back to the training mean.  No plug-in variance then sees the cell's share, and coverage
drops.  The split redraws each sample from its seed and refits nothing.

Replication ``r`` draws the seed ``100_000 + r`` of :func:`tests.discrete_law_longitudinal_sdr.sample`,
the design-probe range that no declared stream reads.  The diagnostic cannot change the cell's
verdict.

The rows go to ``rows.csv.gz``, the reading to ``reading.csv`` and the split to ``strata.csv``.
``tests/unit/test_sdr_below_tail_size_reading.py`` rebuilds the reading and the split.

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


#: The always-path cell, as ``column -> value``.
ALWAYS_PATH = {"W": 1.0, "A1": 1.0, "C1": 1.0, "L2": 1.0, "A2": 1.0, "C2": 1.0}
#: The strata of :func:`strata`, in their published order.
STRATA = ("empty", "one row", "two or more, all Y = 1", "two or more, all Y = 0", "mixed")


def stratum(frame: pd.DataFrame) -> str:
    """What the always-path cell of one draw holds."""
    inside = np.ones(len(frame), dtype=bool)
    for column, value in ALWAYS_PATH.items():
        inside &= frame[column].to_numpy() == value
    outcomes = frame.loc[inside, "Y"].to_numpy()
    if len(outcomes) == 0:
        return STRATA[0]
    if len(outcomes) == 1:
        return STRATA[1]
    if np.all(outcomes == 1.0):
        return STRATA[2]
    if np.all(outcomes == 0.0):
        return STRATA[3]
    return STRATA[4]


def strata(rows: pd.DataFrame) -> pd.DataFrame:
    """Draws and ``SDR`` coverage in each stratum of the always-path cell, redrawn from the seeds."""
    truth = float(law.TRUTH[base.STATIC])
    covered = np.abs(rows["sdr_estimate"] - truth) <= CRITICAL * rows["sdr_std_error"]
    labels = pd.Series(
        [stratum(law.sample(N, SEED_BASE + int(r))) for r in rows["replicate"]], index=rows.index
    )
    out = [
        {
            "stratum": name,
            "draws": int((labels == name).sum()),
            "sdr_coverage": float(covered[labels == name].mean()),
        }
        for name in STRATA
    ]
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
    strata(rows).to_csv(
        arguments.output / "strata.csv", index=False, float_format="%.6g", lineterminator="\n"
    )


if __name__ == "__main__":
    main()
