"""The sizing probe of the X19 studies: second-order bias on fresh draws, before any run.

The five-fold IPSI contrasts carry a second-order bias of order 1/n, so their standardized
bias falls as :math:`1/\\sqrt n`.  This probe measures it once per gated cell at n = 8,000 on
fresh draws, and :mod:`tests.unit.test_longitudinal_incremental_design` projects each value
to the cell's declared size and checks the cell's pass probability there.

It runs, at :data:`N`:

* both S1 primary scenarios, every estimand;
* the S1 calibration fit sets of
  :mod:`tests.studies.longitudinal_incremental_properties` (laws ``A``, ``Ax`` and the mixed
  plans);
* both S2 primary scenarios, every estimand;
* the S2 fit sets of :mod:`tests.studies.longitudinal_incremental_dropout_properties`.

Replication ``r`` of set ``k`` draws the seed ``SEED_BASE + 100_000 k + r``, outside every
declared stream.  The probe cannot change a verdict.  The rows go to ``rows.csv.gz`` and the
reading to ``reading.csv``.

    python -m tests.diagnostics.x19_sizing.run --jobs 4
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_longitudinal_incremental as s1
from tests.studies import canonical_longitudinal_incremental_dropout as s2
from tests.studies import longitudinal_incremental_dropout_properties as p2
from tests.studies import longitudinal_incremental_properties as p1

HERE = Path(__file__).resolve().parent
N = 8_000
REPLICATES = {"s1": 600, "s2": 400}
SEED_BASE = 9_700_000

#: ``(set, study, kind, key)``: a primary scenario or a property fit set.
SETS: tuple[tuple[str, str, str, str], ...] = (
    *(("s1:" + scenario, "s1", "primary", scenario) for scenario in s1.FOLDS),
    ("s1:law_a", "s1", "property", "law_a"),
    ("s1:law_ax", "s1", "property", "law_ax"),
    ("s1:law_a_mixed", "s1", "property", "law_a_mixed"),
    *(("s2:" + scenario, "s2", "primary", scenario) for scenario in s2.SCENARIOS),
    ("s2:law_c", "s2", "property", "law_c"),
    ("s2:law_dc", "s2", "property", "law_dc"),
)


def _seed(index: int, replicate: int) -> int:
    return SEED_BASE + 100_000 * index + replicate


def _fit(index: int, replicate: int) -> list[dict[str, Any]]:
    name, study, kind, key = SETS[index]
    seed = _seed(index, replicate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "primary":
            module = s1 if study == "s1" else s2
            frame, truth = module.draw_from_seed(key, N, seed)
            result = module.fit(frame, key)
            return [
                {
                    "set": name,
                    "cell": estimand,
                    "replicate": replicate,
                    "truth": float(truth[estimand]),
                    "estimate": float(result[estimand].psi),
                    "std_error": float(result[estimand].std_error),
                }
                for estimand in (s1.ESTIMANDS if study == "s1" else s2.ESTIMANDS[key])
            ]
        properties = p1 if study == "s1" else p2
        properties._seed = lambda *_: seed  # type: ignore[attr-defined]
        if study == "s1":
            payload_row = ("interval_calibration", key, "calibration", replicate, N, 1, key)
        else:
            family = "clustered_inference" if key == "law_ck" else "interval_calibration"
            payload_row = (family, key, replicate, N, 1)
        rows = properties._fit_set_rows(payload_row)
    return [
        {
            "set": name,
            "cell": f"{row['property']}/{row['cell']}",
            "replicate": replicate,
            "truth": float(row["truth"]),
            "estimate": float(row["estimate"]),
            "std_error": float(row["std_error"]),
        }
        for row in rows
    ]


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """The standardized bias, its Monte Carlo SE, the SE ratio and the coverage of each cell."""
    out = []
    for (name, cell), group in rows.groupby(["set", "cell"], sort=True):
        error = group["estimate"].to_numpy() - group["truth"].to_numpy()
        spread = float(group["estimate"].std(ddof=1))
        half = 1.959963984540054 * group["std_error"].to_numpy()
        out.append(
            {
                "set": name,
                "cell": cell,
                "n": N,
                "replicates": len(group),
                "standardized_bias": float(error.mean() / spread),
                "mcse": float(1.0 / np.sqrt(len(group))),
                "se_ratio": float(group["std_error"].mean() / spread),
                "coverage": float(np.mean(np.abs(error) <= half)),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    payloads = [
        (index, replicate)
        for index, (_, study, _, _) in enumerate(SETS)
        for replicate in range(REPLICATES[study])
    ]
    outcomes = map_parallel(_fit, payloads, n_jobs=args.jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows.to_csv(HERE / "rows.csv.gz", index=False, lineterminator="\n", float_format="%.17g")
    reading(rows).to_csv(
        HERE / "reading.csv", index=False, lineterminator="\n", float_format="%.6g"
    )


if __name__ == "__main__":
    main()
