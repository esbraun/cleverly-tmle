"""The pilot of the ``ep-crr`` out-of-sieve rescue cell (design review B1 of PR-b).

A pilot, not a result.  On ``d3_complex_limited`` at 1,000 and 4,000 rows it fits EP with the
declared :class:`tests.studies.ep_crr.OffsetOutcome` learner and reads three log ISE ratios
against the oracle: EP, the T-learner at that learner (the control), and EP under the
deliberate mutation that replaces the debiasing weight ``1 / (g Pi)`` by the row weight.  The
cell is gated only if the mutation fails the rescue gate and the control detects.
:func:`tests.studies.ep_crr.pilot_rows` reads the committed summary,
``ep_crr_offset_pilot.csv`` beside this module.

``seconds`` is the CPU time of both EP fits of a replicate; the run fits only the first.

Every draw comes from ``stream_seed(STUDY, "b1_offset_pilot_scale2", label, n, r)``, disjoint from
the run's ``property_sample`` stream and the other pilots::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.ep_crr_offset_pilot --raw <scratch dir> [--summarize]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import ep
from tests.studies import ep_cate
from tests.studies import ep_crr as study
from tests.studies.evidence.seeds import stream_seed

STREAM = "b1_offset_pilot_scale2"
LAW = "d3_complex_limited"
REPLICATES = 31
PILOT_CSV = Path(__file__).with_name("ep_crr_offset_pilot.csv")


#: The EP fit's own reader of the debiasing inputs, which the mutation wraps.
PIECES = ep._pieces


def unit_weight_pieces(fit: Any) -> Any:
    """The mutation: the debiasing weight without ``1 / (g Pi)``."""
    pieces = PIECES(fit)
    return replace(pieces, weight=pieces.rows.copy())


def one(n: int, replicate: int) -> dict[str, Any]:
    law = study.LAW_BY_LABEL[LAW]
    rng = np.random.default_rng(stream_seed(study.STUDY, STREAM, LAW, n, replicate))
    frame = study.sample(law, n, rng)
    start = time.process_time()
    fit = study.fit_ep(law, frame, "offset", replicate, size=ep_cate.schedule)
    right = study.curves(law, frame, fit)
    ep._pieces = unit_weight_pieces  # type: ignore[assignment]
    try:
        mutated = study.curves(
            law, frame, study.fit_ep(law, frame, "offset", replicate, size=ep_cate.schedule)
        )
    finally:
        ep._pieces = PIECES  # type: ignore[assignment]
    return {
        "n": n,
        "r": replicate,
        "ep": math.log(right["ep"] / right["oracle"]),
        "t": math.log(right["t_learner"] / right["oracle"]),
        "unit": math.log(mutated["ep"] / mutated["oracle"]),
        "pd": right["oracle_pd"],
        "mu_l2": right["mu_l2"],
        "seconds": time.process_time() - start,
    }


def summarize(raw: Path) -> pd.DataFrame:
    """One row per group and statistic, in the shape of ``ep_crr_pilot.csv``."""
    frame = pd.read_json(raw, lines=True)
    names = {"ep": "ep_offset", "t": "t_offset", "unit": "ep_offset_unit_weight"}
    rows = []
    for n, group in frame.groupby("n"):
        label = ep_cate.group_label(LAW, int(n))
        for column, statistic in (*names.items(), ("seconds", "offset_seconds")):
            values = group[column].to_numpy(dtype=float)
            rows.append(
                {
                    "group": label,
                    "statistic": statistic,
                    "replicates": values.size,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="the jsonl of replicates")
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--n", type=int, nargs="*", default=[4_000, 1_000])
    parser.add_argument("--summarize", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.summarize:
        summarize(arguments.raw).to_csv(PILOT_CSV, index=False, float_format="%.6g")
        return 0
    done = set()
    if arguments.raw.exists():
        records = [json.loads(line) for line in arguments.raw.read_text().splitlines()]
        done = {(record["n"], record["r"]) for record in records}
    with arguments.raw.open("a") as handle:
        for replicate in range(arguments.replicates):
            for n in arguments.n:
                if (n, replicate) not in done:
                    handle.write(json.dumps(one(n, replicate)) + "\n")
                    handle.flush()
    return 0


__all__ = ["PILOT_CSV", "REPLICATES", "STREAM"]

if __name__ == "__main__":
    raise SystemExit(main())
