r"""Rule 3 of the X26 red-cell policy: the two red cells at n = 4000 with 5,000 replicates.

Rule 2 compared the package curve with two independent curves (``agreement.py``).  With
agreement, rule 3 reruns the red cells at ``n = 4000`` with 5,000 replicates, and rule 8 reads the
result as "conservative, contracting" or "conservative, not contracting at n=4000".

**What is rerun.**  The scenario ``multinomial_logistic_crossfit`` of
``tests.studies.omitted_variable_bound_plugin``: the declared fit, with five-fold cross-fitting,
``q_bounds=(-40, 45)`` and ``g_bounds=(0.005, 0.995)``.  Each replicate draws one sample of 4000
rows and fits it once.  The fit reports the ATE and the ATT of arm 2 against arm 0, and the
lower and upper bound of each.  The cells of the two red results are the lower bounds, and the
upper bounds are read beside them.  The fit, the rebuild of the curve, the reported standard error
and the refusal rule are the study's own (:func:`~tests.studies.omitted_variable_bound_plugin.fit`
and :func:`~tests.studies.omitted_variable_bound_plugin.parameter_rows`).  A refusal of the
package is a lost replicate of that parameter, as in the declared run, and any other error stops
the rerun.

**The seeds.**  Replicate ``k`` draws with ``stream_seed(STUDY, "x26-rule3-rerun", SCENARIO, n,
k)``.  A label hashes into the spawn key, so this stream is disjoint from the primary seeds
(``replicate_seed``) and from the property draws of the declared run (the label
``"property_sample"``).  The size, the count and the seeds are fixed before the first fit.

**Resuming.**  Each replicate writes ``rep-<k>.json`` to the checkpoint directory when it
finishes.  A rerun of the same command fits the replicates that have no file.  The directory holds
``config.json``, which names the size, the label and the parameters.  A directory whose
configuration differs is refused, and ``--fresh`` empties it.  The checkpoints are local scratch and
are not committed.

**The reading.**  ``rerun-rows.csv.gz`` holds one row per cell and surviving replicate, in the
columns of a property replicate.  ``rerun-reading.csv`` is the study's own summary of those rows
(:func:`~tests.studies.omitted_variable_bound_plugin_properties.summarize_properties`): the SE
ratio with its 99% bootstrap interval, the coverage with its 99% interval and the verdict against
the declared calibration bands. The SE-ratio bootstrap uses the study's resampling stream.

**The declared run** passes no ``--replicates`` and no ``--n``::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -m tests.diagnostics.x26_red_cells.rerun --jobs 4

A **smoke run** passes ``--replicates`` below the declared count and an ``--output`` and a
``--checkpoints`` outside the repository.  It writes nothing into this directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

import cleverly
from cleverly.exceptions import DataError
from cleverly.utils.parallel import map_parallel
from tests.studies import omitted_variable_bound_plugin as study
from tests.studies import omitted_variable_bound_plugin_properties as properties
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]

SCENARIO = "multinomial_logistic_crossfit"
#: The parameters of the two red cells.
TOKENS = ("ate_2v0", "att_2v0")
N = 4_000
REPLICATES = 5_000
LABEL = "x26-rule3-rerun"

ROWS_FILE = "rerun-rows.csv.gz"
READING_FILE = "rerun-reading.csv"

#: Where the checkpoints go unless ``--checkpoints`` says otherwise.
DEFAULT_CHECKPOINTS = Path.home() / ".cache" / "cleverly" / "diagnostics" / "x26-red-cells-rerun"


def sample_seed(n: int, replicate: int) -> int:
    """The seed of replicate ``replicate`` at sample size ``n``."""
    return stream_seed(study.STUDY, LABEL, SCENARIO, n, replicate)


def configuration(n: int) -> dict[str, Any]:
    """What a checkpoint directory must agree with before a run reuses it."""
    return {
        "label": LABEL,
        "scenario": SCENARIO,
        "tokens": list(TOKENS),
        "n": n,
        "cleverly_version": cleverly.__version__,
    }


def checkpoint_path(directory: Path, replicate: int) -> Path:
    return directory / f"rep-{replicate:05d}.json"


def fit_replicate(n: int, replicate: int) -> dict[str, Any]:
    """Draw and fit one replicate, and read each parameter's two bounds.

    Returns
    -------
    dict
        ``replicate`` and, for each parameter, either ``refusal`` (the key of the package) or
        the ``lower`` and ``upper`` estimate and reported standard error.
    """
    frame, truth = study.draw_from_seed(SCENARIO, n, sample_seed(n, replicate))
    result = study.fit(frame, SCENARIO)
    out: dict[str, Any] = {"replicate": replicate, "n": n, "parameters": {}}
    for token in TOKENS:
        try:
            values = study.parameter_rows(result, SCENARIO, token)
        except DataError as error:
            out["parameters"][token] = {"refusal": study.refusal_key(str(error))}
            continue
        out["parameters"][token] = {
            row: {
                "truth": float(truth[study.estimand_key(token, row)]),
                "estimate": values.bounds[row].estimate,
                "std_error": values.bounds[row].std_error,
            }
            for row in study.CLAIM_ROWS
        }
    return out


def run_replicate(n: int, replicate: int, directory: Path) -> None:
    """Fit replicate ``replicate`` unless a checkpoint holds it, and write its checkpoint."""
    path = checkpoint_path(directory, replicate)
    if path.exists():
        return
    record = fit_replicate(n, replicate)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temporary, path)


def prepare(directory: Path, n: int, *, fresh: bool) -> None:
    """Create the checkpoint directory, or refuse one that another configuration wrote."""
    wanted = configuration(n)
    if fresh and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "config.json"
    if marker.exists():
        found = json.loads(marker.read_text(encoding="utf-8"))
        if found != wanted:
            raise SystemExit(
                f"refused: {directory} holds checkpoints of {found}, not {wanted}. "
                "Pass --fresh to discard them"
            )
    else:
        marker.write_text(json.dumps(wanted, sort_keys=True), encoding="utf-8")


def collect(directory: Path, replicates: int, n: int) -> pd.DataFrame:
    """The property rows of every surviving replicate in the checkpoints.

    Parameters
    ----------
    directory : Path
        The checkpoint directory.
    replicates : int
        The requested replicates.  Every one must have a checkpoint.
    n : int
        The sample size the checkpoints were fitted at.

    Returns
    -------
    pandas.DataFrame
        One row per cell and surviving replicate, in the columns of a property replicate, with
        ``failed_replicates`` stamped from the lost replicates of the cell.
    """
    missing = [k for k in range(replicates) if not checkpoint_path(directory, k).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} replicates have no checkpoint, the first is {missing[0]}")
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        record = json.loads(checkpoint_path(directory, replicate).read_text(encoding="utf-8"))
        for token, held in record["parameters"].items():
            if "refusal" in held:
                continue
            for row in study.CLAIM_ROWS:
                values = held[row]
                rows.append(
                    {
                        "property": "interval_calibration",
                        "cell": f"{study.cell_label(SCENARIO, token, row)}__correctly_specified",
                        "role": "positive",
                        "replicate": replicate,
                        "n": n,
                        "requested_replicates": replicates,
                        "failed_replicates": 0,
                        "truth": values["truth"],
                        "estimate": values["estimate"],
                        "std_error": values["std_error"],
                        "covered": int(
                            abs(values["estimate"] - values["truth"])
                            <= study.TWO_SIDED * values["std_error"]
                        ),
                        "rejected": int(
                            abs(values["estimate"] / values["std_error"]) > study.TWO_SIDED
                        ),
                    }
                )
    frame = pd.DataFrame(rows, columns=list(REPLICATE_COLUMNS))
    lost = replicates - frame.groupby("cell")["replicate"].nunique()
    frame["failed_replicates"] = frame["cell"].map(lost).astype(int)
    return frame


def refusals(directory: Path, replicates: int) -> pd.DataFrame:
    """The refusals of each parameter by key, from the checkpoints."""
    counts: dict[tuple[str, str], int] = {}
    for replicate in range(replicates):
        record = json.loads(checkpoint_path(directory, replicate).read_text(encoding="utf-8"))
        for token, held in record["parameters"].items():
            if "refusal" in held:
                key = (token, str(held["refusal"]))
                counts[key] = counts.get(key, 0) + 1
    return pd.DataFrame(
        [
            {"parameter": token, "key": key, "lost": count, "rate": count / replicates}
            for (token, key), count in sorted(counts.items())
        ],
        columns=["parameter", "key", "lost", "rate"],
    )


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """The study's own calibration summary of the rerun rows, one row per cell."""
    summary = properties.summarize_properties(rows)
    return summary.sort_values("cell", ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--fresh", action="store_true")
    arguments = parser.parse_args(argv)
    declared = arguments.replicates == REPLICATES and arguments.n == N
    output = arguments.output.resolve()
    if not declared and output.is_relative_to(ROOT):
        raise SystemExit(
            f"refused: a smoke run (--replicates {arguments.replicates}, --n {arguments.n}) "
            f"writes outside the repository, not {output}"
        )
    directory = arguments.checkpoints or DEFAULT_CHECKPOINTS / f"n{arguments.n}"
    if not declared and Path(directory).resolve().is_relative_to(ROOT):
        raise SystemExit("refused: a smoke run keeps its checkpoints outside the repository")
    prepare(directory, arguments.n, fresh=arguments.fresh)
    done = sum(checkpoint_path(directory, k).exists() for k in range(arguments.replicates))
    print(
        f"{done} of {arguments.replicates} replicates are checkpointed in {directory}", flush=True
    )
    pending = [
        (arguments.n, k, directory)
        for k in range(arguments.replicates)
        if not checkpoint_path(directory, k).exists()
    ]
    map_parallel(run_replicate, pending, n_jobs=arguments.jobs)
    rows = collect(directory, arguments.replicates, arguments.n)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / ROWS_FILE, index=False, float_format="%.17g", lineterminator="\n")
    summary = reading(rows)
    summary.to_csv(output / READING_FILE, index=False, float_format="%.6g", lineterminator="\n")
    shown = [
        "cell",
        "replicates",
        "failed_replicates",
        "se_ratio",
        "se_ratio_ci_lower",
        "se_ratio_ci_upper",
        "coverage",
        "coverage_ci_lower",
        "coverage_ci_upper",
        "passed",
    ]
    print(summary.loc[:, shown].to_string(index=False))
    print(refusals(directory, arguments.replicates).to_string(index=False))
    if math.isnan(float(summary["se_ratio"].iloc[0])):
        raise SystemExit("the reading holds no ratio")


if __name__ == "__main__":
    main()
