"""The stacked-classifier parity check of ``vector-mtp``'s unsupplied pair.

``write`` draws three samples of the ordered law at n = 2,000 under the study's primary fit of
``longitudinal_vector_ordered_classifier`` and writes the natural columns, the shifted
components and this package's ``node_ratio``.  ``tests/canonical/vector_mtp_parity.R`` fits
``lmtp``'s own ratios on the same rows.  ``compare`` reads both and reports the largest
absolute difference per plan.  The scenario becomes an exactness witness only if every
difference is below :data:`~tests.studies.canonical_vector_mtp.SMOKE_GATE`.

Usage::

    python -m tests.canonical.vector_mtp_parity write DIR
    docker run ... Rscript /fixture/vector_mtp_parity.R DIR/draws.csv DIR/lmtp.csv
    python -m tests.canonical.vector_mtp_parity compare DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tests.studies import canonical_vector_mtp as study
from tests.studies.evidence.seeds import stream_seed

DRAWS = 3
N = 2_000


def write(directory: Path) -> None:
    scenario = study.ORDERED_CLASSIFIER
    frames = []
    for draw in range(DRAWS):
        frame = study.common.sample_ordered(N, stream_seed(study.STUDY, "parity", draw))
        result = study.fit(frame, scenario)
        sample = frame.copy()
        for name, values in study.pairing_columns(result, scenario).items():
            sample[name] = values
        sample.insert(0, "row", np.arange(N))
        sample.insert(0, "draw", draw)
        frames.append(sample)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(directory / "draws.csv", index=False)


def compare(directory: Path) -> float:
    draws = pd.read_csv(directory / "draws.csv", float_precision="round_trip")
    lmtp = pd.read_csv(directory / "lmtp.csv", float_precision="round_trip")
    worst = 0.0
    for plan in ("natural", "ordered", "full"):
        mine = draws[["draw", "row", f"ratio__{plan}__1"]]
        theirs = lmtp.loc[lmtp["plan"] == plan, ["draw", "row", "lmtp_ratio"]]
        joined = mine.merge(theirs, on=["draw", "row"], validate="one_to_one")
        gap = float(np.max(np.abs(joined[f"ratio__{plan}__1"] - joined["lmtp_ratio"])))
        print(f"{plan}: largest absolute ratio difference {gap:.3g}")
        worst = max(worst, gap)
    verdict = "passes" if worst < study.SMOKE_GATE else "fails"
    print(f"parity {verdict} at {study.SMOKE_GATE:g}")
    return worst


if __name__ == "__main__":
    command, where = sys.argv[1], Path(sys.argv[2])
    if command == "write":
        write(where)
    elif command == "compare":
        compare(where)
    else:
        raise SystemExit("usage: python -m tests.canonical.vector_mtp_parity write|compare DIR")
