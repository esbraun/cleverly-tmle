"""The pilot of the few-cluster floor studies, on its own ``"pilot"`` seed stream (plan 5, M5).

Run before the declaration is final: 400 draws of every cell of one study, through the study's
own property loop and summary, written as ``pilot.csv.gz`` beside the study's declaration,
with the family-wise sizing of each floor family printed
(:func:`tests.studies.few_cluster_floor.family_sizing`). The pilot reads the
``"pilot"`` stream label, so no pilot draw is a property draw. A pilot never changes a declared
rule; it sets each family's R (steps of 400 from 4,000, capped at 6,000) and records the
predicted reds.

.. code-block:: bash

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python -m tests.studies.few_cluster_floor_pilot tmle --output <scratch dir> --jobs 4
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Final

import pandas as pd

from tests.studies.evidence.replication import recording

#: The pilot's draws per cell (plan 5, general rules).
PILOT_DRAWS: Final[int] = 400
MODULES: Final[dict[str, str]] = {
    "tmle": "tests.studies.few_cluster_floor_tmle_properties",
    "ltmle": "tests.studies.few_cluster_floor_ltmle_properties",
}


def run(which: str, output: Path, jobs: int) -> pd.DataFrame:
    """Fit the pilot of one study, write its rows and print its sizing."""
    module = importlib.import_module(MODULES[which])
    with recording("cleverly", witness=module.PROPERTY_WITNESS):
        rows = module.generate_property_rows(n_jobs=jobs, budget=PILOT_DRAWS, stream="pilot")
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "pilot.csv.gz", index=False)
    summary = module.summarize_properties(rows)
    print(module.sizing(summary))
    return rows


def main() -> None:
    """The command line of :func:`run`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=sorted(MODULES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    run(args.study, args.output, args.jobs)


if __name__ == "__main__":
    main()
