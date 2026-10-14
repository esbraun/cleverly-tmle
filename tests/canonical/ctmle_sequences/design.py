"""Write the two declaration inputs of ``canonical-ctmle-sequences``.

The study reads two committed inputs that no replicate writes:

* ``lambdas.csv`` holds one penalty grid per scenario, each on that scenario's law.
  ``python -m tests.canonical.ctmle_sequences.design lambdas`` writes it from the declared grid
  rule in :func:`tests.studies.canonical_ctmle_sequences.primary_grid`.
* ``r_folds.csv.gz`` holds R's selection folds of each ``binary_learners`` replicate.
  ``python -m tests.canonical.ctmle_sequences.design folds`` writes the replication seeds and
  runs ``make_r_folds.R`` in the reference image on them.  ``seeds --output PATH`` writes only
  the seed table, for a manual container run.

Both inputs are committed before the first run, and both are hashed into the run's manifest.
``power_probe.py`` beside this module writes the pilot table ``power-probe.csv`` (plan 3.4.3).
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from tests.canonical.ctmle_sequences.regenerate import REFERENCE
from tests.canonical.regenerate import write_seeds
from tests.studies import canonical_ctmle_sequences as study
from tests.studies.evidence.manifest import write_csv

HERE = Path(__file__).resolve().parent
RUNNER_STEM = Path(REFERENCE.runner).stem


def write_lambdas(path: Path) -> None:
    """Write one grid per scenario, one penalty per row, at full precision.

    Parameters
    ----------
    path : Path
        The CSV to write.
    """
    frames = []
    for scenario in study.SCENARIOS:
        grid = study.primary_grid(scenario)
        frames.append(
            pd.DataFrame({"scenario": scenario, "entry": range(1, len(grid) + 1), "lambda": grid})
        )
    write_csv(pd.concat(frames, ignore_index=True), path, float_format="%.17g")


def write_folds(path: Path, *, image: str) -> None:
    """Draw R's ``binary_learners`` folds in ``image`` and write them to ``path``.

    Parameters
    ----------
    path : Path
        The gzipped CSV to write, inside this directory.
    image : str
        The reference image that holds R.
    """
    with tempfile.TemporaryDirectory() as scratch:
        seeds = Path(scratch) / "seeds.csv"
        write_seeds(seeds, study.STUDY, RUNNER_STEM)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "Rscript",
                "-v",
                f"{HERE}:/fixture:ro",
                "-v",
                f"{scratch}:/work",
                image,
                "/fixture/make_r_folds.R",
                "/work/seeds.csv",
                str(study.PRIMARY_N),
                "/work/r_folds.csv.gz",
            ],
            check=True,
        )
        path.write_bytes((Path(scratch) / "r_folds.csv.gz").read_bytes())


def main() -> None:
    """Run one subcommand: ``lambdas``, ``seeds`` or ``folds``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    lambdas = commands.add_parser("lambdas", help="write lambdas.csv")
    lambdas.add_argument("--output", type=Path, default=study.LAMBDAS_FILE)
    seeds = commands.add_parser("seeds", help="write the replication seeds of the R runner")
    seeds.add_argument("--output", type=Path, required=True)
    folds = commands.add_parser("folds", help="write r_folds.csv.gz in the reference image")
    folds.add_argument("--output", type=Path, default=study.R_FOLDS_FILE)
    folds.add_argument("--image", default=REFERENCE.image)
    arguments = parser.parse_args()
    if arguments.command == "lambdas":
        write_lambdas(arguments.output)
    elif arguments.command == "seeds":
        write_seeds(arguments.output, study.STUDY, RUNNER_STEM)
    else:
        write_folds(arguments.output, image=arguments.image)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
