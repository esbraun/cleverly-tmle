"""The SE-ratio pilot of the ``learned-rule-ep-declared`` study (plan 5.2; design review S2).

A pilot, not a result: it fixes the SE ratios that each arm's calibration ``R`` is computed
from, before the declared run.  Each gated calibration cell is fitted :data:`REPLICATES` times
on the ``pilot`` stream of :data:`tests.studies.learned_rule_ep_declared.STREAMS`, whose sample
labels (``pilot_sample``) and fold labels (``pilot_fold_partition``) are disjoint from the
declared run's, through the declared run's own row path
(:func:`~tests.studies.learned_rule_ep_declared_properties.generate_property_rows`).  Every
pilot replicate refits and compares its fold rules (:func:`~tests.studies.learned_rule_ep_declared.verified`).

The committed output is :data:`PILOT_CSV`, one row per gated cell with its SE ratio, the
ratio's bootstrap standard error, the coverage and the standardized bias.  Every cell draws its
own samples, so the cells' errors are independent and the label correlations are the identity.
:func:`declared_replicates` returns X32a's ``R`` of each arm
(:func:`tests.studies.repeated_sizing.calibration_replicates`), which the declaration commit
writes into :data:`tests.studies.learned_rule_ep_declared.BUDGETED_R` where it is larger.  The
pilot is written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.learned_rule_ep_declared_pilot --raw <scratch dir> --jobs 16

which keeps the raw rows in ``<scratch dir>/pilot-rows.parquet`` and writes the CSV.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import learned_rule_ep_declared as study
from tests.studies import learned_rule_ep_declared_properties as properties
from tests.studies import repeated_sizing
from tests.studies.evidence.seeds import stream_seed

#: The stream the pilot draws from.
STREAM = "pilot"

#: Draws of each gated cell (plan 5.2).
REPLICATES = study.PILOT_DRAWS

#: Bootstrap resamples of each ratio's standard error.
BOOTSTRAP = 400

#: The committed summary.
PILOT_CSV = Path(__file__).with_name("learned_rule_ep_declared_pilot.csv")

COLUMNS = ("cell", "arm", "replicates", "se_ratio", "se_ratio_se", "coverage", "standardized_bias")


def pilot_rows(
    *, replicates: int = REPLICATES, n_jobs: int = STUDY_JOBS, cells: Sequence[str] | None = None
) -> pd.DataFrame:
    """Fit the pilot draws of each gated cell on the pilot stream."""
    return properties.generate_property_rows(
        n_jobs=n_jobs,
        budget=replicates,
        stream=STREAM,
        cells=tuple(cells or study.POSITIVE_CELLS),
    )


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """The per-cell pilot summary of the gated cells."""
    summary = []
    for cell in study.POSITIVE_CELLS:
        group = rows.loc[(rows["cell"] == cell) & (rows["role"] == "positive")]
        if group.empty:
            continue
        error = group["estimate"].to_numpy(float) - group["truth"].to_numpy(float)
        se = group["std_error"].to_numpy(float)
        sd = float(np.std(error, ddof=1))
        rng = np.random.default_rng(stream_seed(study.STUDY, STREAM, "bootstrap", cell))
        index = rng.integers(0, len(group), (BOOTSTRAP, len(group)))
        resampled = se[index].mean(axis=1) / error[index].std(axis=1, ddof=1)
        summary.append(
            {
                "cell": cell,
                "arm": f"__{cell.rsplit('__', 1)[-1]}",
                "replicates": len(group),
                "se_ratio": float(se.mean() / sd),
                "se_ratio_se": float(resampled.std(ddof=1)),
                "coverage": float(group["covered"].mean()),
                "standardized_bias": float(np.mean(error) / sd),
            }
        )
    return pd.DataFrame(summary, columns=list(COLUMNS))


def independent(labels: Sequence[str]) -> np.ndarray:
    """The label correlations: every cell draws its own samples."""
    return np.eye(len(labels))


def declared_replicates(summary: pd.DataFrame) -> dict[str, int]:
    """X32a's calibration ``R`` of each arm from the pilot."""
    out = {}
    for arm, group in summary.groupby("arm"):
        out[str(arm).removeprefix("__")] = repeated_sizing.calibration_replicates(
            study.STUDY.margins,
            dict(zip(group["cell"], group["se_ratio"], strict=True)),
            int(group["replicates"].min()),
            independent,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--cells", nargs="*", default=None)
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    rows = pilot_rows(replicates=arguments.replicates, n_jobs=arguments.jobs, cells=arguments.cells)
    rows.to_parquet(arguments.raw / "pilot-rows.parquet")
    summary = summarize(rows)
    summary.to_csv(PILOT_CSV, index=False, float_format="%.6g")
    print(f"wrote {PILOT_CSV}", flush=True)
    print(f"calibration R from the pilot: {declared_replicates(summary)}", flush=True)


if __name__ == "__main__":
    main()
