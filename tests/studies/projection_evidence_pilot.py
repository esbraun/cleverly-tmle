"""The pilot of the ``projection-evidence`` study (plan rev 3, 5.0 and 5.4), declared before it runs.

A pilot, not a result: it fixes the SE ratios the calibration ``R`` is computed from and the
joint coverage each band group is read at, before the declared run.  Each pilot group
(:func:`tests.studies.projection_evidence.pilot_groups`) is fitted :data:`REPLICATES` times on the
``pilot`` stream through the declared run's own row path.  Every arm whose plug-in pass at the
budget (:func:`arm_pass`) reads below
:data:`~tests.studies.projection_evidence.PILOT_2_BELOW` is drawn again on the ``pilot_2`` stream
(:data:`~tests.studies.projection_evidence.PILOT_2_DRAWS` more), and the two are pooled.

The committed outputs are :data:`PILOT_CSV` (one row per positive calibration cell: SE ratio, its
bootstrap SE, coverage, standardized bias), :data:`CORRELATIONS_CSV` (the error correlation of
each pair of cells that share a group's fits) and :data:`BANDS_CSV` (the joint coverage of each
band and pointwise control).  :func:`declared_replicates` reads them and returns X32a's ``R`` over
every calibration arm with one heterogeneity spread; if no ``R`` up to 50,000 meets it, the
point-reading fallback (:mod:`tests.studies.point_reading`) at
:data:`~tests.studies.projection_evidence.POINT_READING_TARGET`.  The pilot is written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.projection_evidence_pilot --raw <scratch dir> --jobs 16

and ``--stream pilot_2 --groups <labels>`` for the extension.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import point_reading, repeated_sizing
from tests.studies import projection_evidence as study
from tests.studies import projection_evidence_properties as properties
from tests.studies.evidence.seeds import stream_seed

#: Draws of each pilot group, and of its ``pilot_2`` extension.
REPLICATES = study.PILOT_DRAWS
EXTENSION = study.PILOT_2_DRAWS

#: Bootstrap resamples of each ratio's standard error.
BOOTSTRAP = 400

#: The committed summaries.
PILOT_CSV = Path(__file__).with_name("projection_evidence_pilot.csv")
CORRELATIONS_CSV = Path(__file__).with_name("projection_evidence_pilot_correlations.csv")
BANDS_CSV = Path(__file__).with_name("projection_evidence_pilot_bands.csv")

COLUMNS = (
    "cell",
    "group",
    "arm",
    "replicates",
    "se_ratio",
    "se_ratio_se",
    "coverage",
    "covered",
    "standardized_bias",
)
BAND_COLUMNS = ("cell", "group", "replicates", "coverage")


def positive_cells() -> tuple[str, ...]:
    """Every positive calibration cell, in arm order."""
    return tuple(cell for cells in study.CALIBRATION_ARMS.values() for cell in cells)


def arm_of(cell: str) -> str:
    """The calibration arm of a positive cell."""
    return next(arm for arm, cells in study.CALIBRATION_ARMS.items() if cell in cells)


def group_of(cell: str) -> str:
    """The group of a cell, by its label prefix."""
    return max(
        (group.label for group in study.GROUPS if cell.startswith(f"{group.label}_")), key=len
    )


def pilot_rows(
    *,
    replicates: int = REPLICATES,
    n_jobs: int = STUDY_JOBS,
    groups: Sequence[str] | None = None,
    stream: str = "pilot",
) -> pd.DataFrame:
    """Fit the pilot draws of each pilot group on ``stream``."""
    return properties.generate_property_rows(
        n_jobs=n_jobs,
        budget=replicates,
        groups=tuple(groups or study.pilot_groups()),
        stream=stream,
    )


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The calibration summary, the error correlations of cells that share fits, and the bands."""
    positives = rows.loc[rows["cell"].isin(positive_cells()) & (rows["role"] == "positive")]
    summary = []
    errors: dict[str, pd.Series] = {}
    for cell in positive_cells():
        cell_rows = positives.loc[positives["cell"] == cell]
        if cell_rows.empty:
            continue
        estimate = cell_rows["estimate"].to_numpy(float)
        se = cell_rows["std_error"].to_numpy(float)
        truth = cell_rows["truth"].to_numpy(float)
        sd = float(np.std(estimate - truth, ddof=1))
        rng = np.random.default_rng(stream_seed(study.STUDY, "pilot", "bootstrap", cell))
        index = rng.integers(0, len(cell_rows), (BOOTSTRAP, len(cell_rows)))
        resampled = se[index].mean(axis=1) / (estimate - truth)[index].std(axis=1, ddof=1)
        summary.append(
            {
                "cell": cell,
                "group": group_of(cell),
                "arm": arm_of(cell),
                "replicates": len(cell_rows),
                "se_ratio": float(se.mean() / sd),
                "se_ratio_se": float(resampled.std(ddof=1)),
                "coverage": float(cell_rows["covered"].mean()),
                "covered": int(cell_rows["covered"].sum()),
                "standardized_bias": float(np.mean(estimate - truth) / sd),
            }
        )
        errors[cell] = pd.Series(estimate - truth, index=cell_rows["replicate"].to_numpy())
    pairs = []
    for first, second in itertools.combinations(errors, 2):
        if group_of(first) != group_of(second):
            continue
        joined = pd.concat([errors[first], errors[second]], axis=1, join="inner")
        pairs.append(
            {
                "first": first,
                "second": second,
                "correlation": float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1]),
            }
        )
    joint = rows.loc[rows["property"] == study.SIMULTANEOUS]
    bands = [
        {
            "cell": cell,
            "group": group_of(cell),
            "replicates": len(cell_rows),
            "coverage": float(cell_rows["covered"].mean()),
        }
        for cell, cell_rows in joint.groupby("cell", sort=False)
    ]
    return (
        pd.DataFrame(summary, columns=list(COLUMNS)),
        pd.DataFrame(pairs, columns=["first", "second", "correlation"]),
        pd.DataFrame(bands, columns=list(BAND_COLUMNS)),
    )


def arm_pass(summary: pd.DataFrame, replicates: int = study.BUDGETED_R) -> dict[str, float]:
    """Each arm's plug-in pass at ``replicates``: its rows read at their pilot ratios."""
    margins = study.STUDY.margins
    return {
        str(arm): float(
            np.prod(
                repeated_sizing.calibration_pass(margins, replicates, rows["se_ratio"].to_numpy())
            )
        )
        for arm, rows in summary.groupby("arm", sort=False)
    }


def pilot_2_arms(summary: pd.DataFrame) -> tuple[str, ...]:
    """The arms the ``pilot_2`` extension draws: plug-in pass below the declared floor."""
    return tuple(arm for arm, p in arm_pass(summary).items() if p < study.PILOT_2_BELOW)


def label_correlations(pairs: pd.DataFrame) -> repeated_sizing.Correlations:
    """The label-correlation function: the pilot's error correlation within a group, else 0."""
    table: dict[tuple[str, str], float] = {}
    for row in pairs.itertuples(index=False):
        first, second = str(row.first).split("__", 1)[0], str(row.second).split("__", 1)[0]
        table[first, second] = table[second, first] = float(row.correlation)

    def correlations(labels: Sequence[str]) -> np.ndarray:
        return np.array(
            [[1.0 if a == b else table.get((a, b), 0.0) for b in labels] for a in labels]
        )

    return correlations


def x32a_replicates(summary: pd.DataFrame, pairs: pd.DataFrame) -> int:
    """X32a's ``R`` over every calibration arm, one heterogeneity spread (plan 5.0 rule 2)."""
    ratios: Mapping[str, float] = dict(zip(summary["cell"], summary["se_ratio"], strict=True))
    return repeated_sizing.calibration_replicates(
        study.STUDY.margins,
        ratios,
        int(summary["replicates"].min()),
        label_correlations(pairs),
    )


#: The message of :func:`repeated_sizing.calibration_replicates` when no budget meets X32a's
#: rule: the only error the point-reading fallback may answer.
NO_BUDGET = "no budget up to 50,000 replications meets the calibration rule"


def pilot_2_shortfall(summary: pd.DataFrame) -> list[str]:
    """The arms ``pilot_2`` should have extended that hold fewer than both pilots' draws."""
    needed = study.PILOT_DRAWS + study.PILOT_2_DRAWS
    rows = summary
    short = []
    for arm in pilot_2_arms(summary):
        fewest = int(rows.loc[rows["arm"] == arm, "replicates"].min())
        if fewest < needed:
            short.append(f"{arm}: {fewest} draws, below {needed}")
    return short


def declared_replicates(
    summary: pd.DataFrame, pairs: pd.DataFrame, *, diagnosed: Collection[str] = ()
) -> tuple[int, str]:
    """The calibration ``R`` and the rule that set it: ``"x32a"`` or ``"point_reading"``.

    X32a's rule first; if no ``R`` up to 50,000 meets it, and every arm ``pilot_2`` extends
    holds both pilots' draws, the point-reading fallback at
    :data:`~tests.studies.projection_evidence.POINT_READING_TARGET`, which refuses unless every
    row whose 99% interval excludes 1 is in ``diagnosed`` and lies within the absorb band.
    The declared ``R`` is never below :data:`~tests.studies.projection_evidence.BUDGETED_R`.
    """
    try:
        computed, rule = x32a_replicates(summary, pairs), "x32a"
    except ValueError as error:
        if NO_BUDGET not in str(error):
            raise
        short = pilot_2_shortfall(summary)
        if short:
            raise ValueError(
                "the point-reading fallback runs only after pilot_2: " + "; ".join(short)
            ) from error
        computed = point_reading.point_reading_replicates(
            study.STUDY.margins,
            dict(zip(summary["cell"], summary["se_ratio"], strict=True)),
            dict(zip(summary["cell"], summary["se_ratio_se"], strict=True)),
            target=study.POINT_READING_TARGET,
            diagnosed=diagnosed,
        )
        rule = "point_reading"
    return max(computed, study.BUDGETED_R), rule


def band_replicates(bands: pd.DataFrame) -> dict[str, int]:
    """Each gated band group's ``R``: rebuilt at 4,000 where its pilot joint coverage is low."""
    out = {}
    for row in bands.itertuples(index=False):
        if not str(row.cell).endswith(f"__{study.BAND_KIND}"):
            continue
        group = study.GROUP_BY_LABEL[str(row.group)]
        if group.band != "gated":
            continue
        low = float(row.coverage) < study.BAND_REBUILD_BELOW
        out[group.label] = study.BAND_REBUILT_R if low else study.BAND_R
    return out


def primary_replicates(summary: pd.DataFrame) -> int:
    """The primary ``R`` from the pilot's covered counts of the ``cde_4k`` intercepts (plan 5.4)."""
    group = study.GROUP_BY_LABEL[study.PRIMARY_GROUP]
    cells = [study.calibration_cell(group, arm, "(intercept)") for arm in group.arms]
    rows = summary.set_index("cell").loc[cells]
    return repeated_sizing.primary_replicates(
        study.STUDY.margins,
        dict(zip(cells, rows["covered"].astype(int), strict=True)),
        int(rows["replicates"].min()),
        implementations=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--stream", default="pilot", choices=("pilot", "pilot_2"))
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    rows = pilot_rows(
        replicates=arguments.replicates,
        n_jobs=arguments.jobs,
        groups=arguments.groups,
        stream=arguments.stream,
    )
    if arguments.stream == "pilot_2":
        # The extension's replicates follow the pilot's, so pooled rows keep distinct indices.
        rows["replicate"] += REPLICATES
    rows.to_parquet(arguments.raw / f"{arguments.stream}-rows.parquet")
    pooled = [pd.read_parquet(path) for path in sorted(arguments.raw.glob("pilot*-rows.parquet"))]
    summary, pairs, bands = summarize(pd.concat(pooled, ignore_index=True))
    summary.to_csv(PILOT_CSV, index=False, float_format="%.6g")
    pairs.to_csv(CORRELATIONS_CSV, index=False, float_format="%.6g")
    bands.to_csv(BANDS_CSV, index=False, float_format="%.6g")
    print(f"pilot_2 arms: {pilot_2_arms(summary)}", flush=True)
    print(f"band R: {band_replicates(bands)}", flush=True)
    print(f"primary R: {primary_replicates(summary)}", flush=True)
    print(f"calibration R: {declared_replicates(summary, pairs)}", flush=True)


if __name__ == "__main__":
    main()
