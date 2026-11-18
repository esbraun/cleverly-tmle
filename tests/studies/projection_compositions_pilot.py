"""The SE-ratio pilot of the ``projection-compositions`` study (plan 5.0; design review S2).

A pilot, not a result: it fixes the SE ratios that the calibration ``R`` is computed from,
before the declared run.  Each gated calibration group is fitted :data:`REPLICATES` times on the
``pilot`` stream of :data:`tests.studies.projection_compositions_properties.STREAMS`, whose sample
labels (``pilot_sample``) and fold labels (``pilot_fold_partition``) are disjoint from the
declared run's ``property_sample`` and ``fold_partition``, through the declared run's own row
path (:func:`~tests.studies.projection_compositions_properties.generate_property_rows`).

The committed outputs are :data:`PILOT_CSV`, one row per positive calibration cell with its SE
ratio, the ratio's bootstrap standard error, the coverage and the standardized bias, and
:data:`CORRELATIONS_CSV`, the correlation of the errors ``estimate - truth`` of every pair of
positive cells that share a group's fits.  Cells of different groups share no draw, so their
correlation is zero.  :func:`declared_replicates` reads both and returns X32a's ``R``
(:func:`tests.studies.repeated_sizing.calibration_replicates`), which the declaration commit
writes into :data:`tests.studies.projection_compositions.BUDGETED_R` when it is larger.  The
pilot is written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.projection_compositions_pilot --raw <scratch dir> --jobs 16

which keeps the raw rows in ``<scratch dir>/pilot-rows.parquet`` and writes both CSVs.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import projection_compositions as study
from tests.studies import projection_compositions_properties as properties
from tests.studies import repeated_sizing
from tests.studies.evidence.seeds import stream_seed

#: The stream the pilot draws from.
STREAM = "pilot"

#: Draws of each gated group (plan 5.0).
REPLICATES = study.PILOT_DRAWS

#: Bootstrap resamples of each ratio's standard error.
BOOTSTRAP = 400

#: The committed summaries.
PILOT_CSV = Path(__file__).with_name("projection_compositions_pilot.csv")
CORRELATIONS_CSV = Path(__file__).with_name("projection_compositions_pilot_correlations.csv")

COLUMNS = (
    "cell",
    "group",
    "replicates",
    "se_ratio",
    "se_ratio_se",
    "coverage",
    "standardized_bias",
)


def pilot_groups() -> tuple[str, ...]:
    """The gated calibration groups, in declaration order."""
    return tuple(group.label for group in study.GROUPS if group.arms and not group.reporting)


def positive_cells() -> tuple[str, ...]:
    """Every positive calibration cell, in arm order."""
    return tuple(cell for cells in study.CALIBRATION_ARMS.values() for cell in cells)


def pilot_rows(
    *, replicates: int = REPLICATES, n_jobs: int = STUDY_JOBS, groups: Sequence[str] | None = None
) -> pd.DataFrame:
    """Fit the pilot draws of each gated group on the pilot stream."""
    return properties.generate_property_rows(
        n_jobs=n_jobs,
        budget=replicates,
        groups=tuple(groups or pilot_groups()),
        stream=STREAM,
    )


def _group_of(cell: str) -> str:
    return next(group.label for group in study.GROUPS if cell.startswith(f"{group.label}_"))


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-cell pilot summary and the error correlations of cells that share fits."""
    positives = rows.loc[rows["cell"].isin(positive_cells()) & (rows["role"] == "positive")]
    summary = []
    errors: dict[str, pd.Series] = {}
    for cell in positive_cells():
        group = positives.loc[positives["cell"] == cell]
        if group.empty:
            continue
        estimate = group["estimate"].to_numpy(float)
        se = group["std_error"].to_numpy(float)
        truth = group["truth"].to_numpy(float)
        sd = float(np.std(estimate - truth, ddof=1))
        rng = np.random.default_rng(stream_seed(study.STUDY, STREAM, "bootstrap", cell))
        index = rng.integers(0, len(group), (BOOTSTRAP, len(group)))
        resampled = se[index].mean(axis=1) / (estimate - truth)[index].std(axis=1, ddof=1)
        summary.append(
            {
                "cell": cell,
                "group": _group_of(cell),
                "replicates": len(group),
                "se_ratio": float(se.mean() / sd),
                "se_ratio_se": float(resampled.std(ddof=1)),
                "coverage": float(group["covered"].mean()),
                "standardized_bias": float(np.mean(estimate - truth) / sd),
            }
        )
        errors[cell] = pd.Series(estimate - truth, index=group["replicate"].to_numpy())
    pairs = []
    for first, second in itertools.combinations(errors, 2):
        if _group_of(first) != _group_of(second):
            continue
        joined = pd.concat([errors[first], errors[second]], axis=1, join="inner")
        pairs.append(
            {
                "first": first,
                "second": second,
                "correlation": float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]),
            }
        )
    return (
        pd.DataFrame(summary, columns=list(COLUMNS)),
        pd.DataFrame(pairs, columns=["first", "second", "correlation"]),
    )


def label_correlations(pairs: pd.DataFrame) -> repeated_sizing.Correlations:
    """The label-correlation function of :func:`repeated_sizing.calibration_replicates`.

    A cell's label is its name before ``__``.  Two cells of one group take the pilot's error
    correlation; cells of different groups share no draw and take zero.
    """
    table: dict[tuple[str, str], float] = {}
    for row in pairs.itertuples(index=False):
        first, second = str(row.first).split("__", 1)[0], str(row.second).split("__", 1)[0]
        table[first, second] = table[second, first] = float(row.correlation)

    def correlations(labels: Sequence[str]) -> np.ndarray:
        return np.array(
            [[1.0 if a == b else table.get((a, b), 0.0) for b in labels] for a in labels]
        )

    return correlations


def declared_replicates(summary: pd.DataFrame, pairs: pd.DataFrame) -> int:
    """X32a's calibration ``R`` from the pilot: the smallest budget step with family pass 0.95."""
    ratios: Mapping[str, float] = dict(zip(summary["cell"], summary["se_ratio"], strict=True))
    return repeated_sizing.calibration_replicates(
        study.STUDY.margins,
        ratios,
        int(summary["replicates"].min()),
        label_correlations(pairs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--groups", nargs="*", default=None)
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    rows = pilot_rows(
        replicates=arguments.replicates, n_jobs=arguments.jobs, groups=arguments.groups
    )
    rows.to_parquet(arguments.raw / "pilot-rows.parquet")
    summary, pairs = summarize(rows)
    summary.to_csv(PILOT_CSV, index=False, float_format="%.6g")
    pairs.to_csv(CORRELATIONS_CSV, index=False, float_format="%.6g")
    print(f"wrote {PILOT_CSV} and {CORRELATIONS_CSV}", flush=True)
    print(f"calibration R from the pilot: {declared_replicates(summary, pairs)}", flush=True)


if __name__ == "__main__":
    main()
