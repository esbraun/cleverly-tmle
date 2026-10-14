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
* the S2 fit sets of :mod:`tests.studies.longitudinal_incremental_dropout_properties`;
* the competing-risk fit set again at its declared size, :data:`COMPETING_N`;
* at its declared size, each cell whose rule the bias projection does not reach: the sharp
  null, the power cell, both ``single_robustness`` sizes, both bands with the S2 survival
  calibration and one-fold cells they carry, and the clustered pair.

The competing-risk cell needs its own size.  At n = 8,000 its SE ratio reads 0.85 on 1,000
draws while its coverage reads 0.95.  About 0.6% of draws carry a reported SE 3 to 14 times
the median, and those draws dominate the mean SE.  Each of them has one history cell of about
13 units in which a training fold holds one or two treated units.  The saturated cell-mean
fit then puts a censoring probability at the lower bound, 0.01.  The reported SE tracks the
error on those draws, so this is a finite-sample tail and not a defect in the curve.  It does
not shrink as :math:`1/n`, so the design test reads that cell at its declared size and does
not project it.

Replication ``r`` of set ``k`` draws the seed ``SEED_BASE + 100_000 k + r``, outside every
declared stream.  The probe cannot change a verdict.  The rows go to ``rows.csv.gz`` and the
reading to ``reading.csv``.

    python -m tests.diagnostics.x19_sizing.run --jobs 4
    python -m tests.diagnostics.x19_sizing.run --jobs 4 --set s2:law_dc@32000
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
COMPETING_N = p2.COMPETING_N
SEED_BASE = 9_700_000

#: ``set: (index, study, kind, key, n, replicates)``: a primary scenario or a property fit set.
#: The index fixes the seed stream.
SETS: dict[str, tuple[int, str, str, str, int, int]] = {
    "s1:ipsi_folds_2": (0, "s1", "primary", "ipsi_folds_2", N, 600),
    "s1:ipsi_folds_5": (1, "s1", "primary", "ipsi_folds_5", N, 600),
    "s1:law_a": (2, "s1", "property", "law_a", N, 600),
    "s1:law_ax": (3, "s1", "property", "law_ax", N, 600),
    "s1:law_a_mixed": (4, "s1", "property", "law_a_mixed", N, 600),
    "s2:dropout_folds_5": (5, "s2", "primary", "dropout_folds_5", N, 400),
    "s2:survival_folds_5": (6, "s2", "primary", "survival_folds_5", N, 400),
    "s2:law_c": (7, "s2", "property", "law_c", N, 400),
    "s2:law_dc": (8, "s2", "property", "law_dc", N, 400),
    f"s2:law_dc@{COMPETING_N}": (9, "s2", "property", "law_dc", COMPETING_N, 1_000),
    # The cells the bias projection does not reach, each read at its declared size.
    "s1:sharp_null": (10, "s1", "fit_set", "sharp_null", p1.NULL_N, 1_000),
    "s1:alternative": (11, "s1", "fit_set", "alternative", p1.NULL_N, 1_000),
    **{
        f"s1:outcome_linear__n_{size}": (
            index,
            "s1",
            "fit_set",
            f"outcome_linear__n_{size}",
            size,
            1_000,
        )
        for index, size in enumerate(p1.REQUIREMENT_SIZES, start=12)
    },
    "s1:band": (14, "s1", "band", p1.BAND_LABEL, p1.BAND_N, 1_000),
    "s2:band": (15, "s2", "band", p2.BAND_LABEL, p2.N, 1_000),
    "s2:law_ck": (16, "s2", "property", "law_ck", p2.CLUSTER_N, 1_000),
    "s2:dropout_folds_1": (17, "s2", "primary", "dropout_folds_1", N, 600),
    "s2:survival_folds_1": (18, "s2", "primary", "survival_folds_1", N, 600),
    # At 32,000 the tail is gone and the SE ratio reads 0.948 (MCSE about 0.02), so a second
    # reading at twice the size fixes the decay of the remaining deficit.
    "s2:law_dc@64000": (19, "s2", "property", "law_dc", 64_000, 1_000),
}


def _seed(index: int, replicate: int) -> int:
    return SEED_BASE + 100_000 * index + replicate


def _fit(name: str, replicate: int) -> list[dict[str, Any]]:
    index, study, kind, key, n, _ = SETS[name]
    seed = _seed(index, replicate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "primary":
            module = s1 if study == "s1" else s2
            frame, truth = module.draw_from_seed(key, n, seed)
            result = module.fit(frame, key)
            return [
                {
                    "set": name,
                    "cell": estimand,
                    "n": n,
                    "replicate": replicate,
                    "truth": float(truth[estimand]),
                    "estimate": float(result[estimand].psi),
                    "std_error": float(result[estimand].std_error),
                }
                for estimand in (s1.ESTIMANDS if study == "s1" else s2.ESTIMANDS[key])
            ]
        properties = p1 if study == "s1" else p2
        properties._seed = lambda *_: seed  # type: ignore[attr-defined]
        if kind == "band":
            rows = properties._band_rows((replicate, 1))
        elif kind == "fit_set":
            (family, label, configuration, _, _, law_name) = next(
                entry for entry in p1.FIT_SETS if entry[1] == key
            )
            rows = p1._fit_set_rows((family, label, configuration, replicate, n, 1, law_name))
        elif study == "s1":
            rows = p1._fit_set_rows(
                ("interval_calibration", key, "calibration", replicate, n, 1, key)
            )
        else:
            family = "clustered_inference" if key == "law_ck" else "interval_calibration"
            rows = p2._fit_set_rows((family, key, replicate, n, 1))
    return [
        {
            "set": name,
            "cell": f"{row['property']}/{row['cell']}",
            "n": n,
            "replicate": replicate,
            "truth": float(row["truth"]),
            "estimate": float(row["estimate"]),
            "std_error": float(row["std_error"]),
            "covered": float(row["covered"]),
            "rejected": float(row["rejected"]),
        }
        for row in rows
    ]


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """Each cell's standardized bias, Monte Carlo SE, SE ratio and coverage.

    ``covered`` and ``rejected`` are the rates the study rows record, which a joint-coverage
    row reads.  ``bias`` and ``bias_se`` are on the estimand's own scale, which the absolute
    rule of ``single_robustness`` reads.  The first probe recorded neither rate, so they are
    empty on its sets.
    """
    out = []
    for (name, cell), group in rows.groupby(["set", "cell"], sort=True):
        error = group["estimate"].to_numpy() - group["truth"].to_numpy()
        spread = float(group["estimate"].std(ddof=1))
        half = 1.959963984540054 * group["std_error"].to_numpy()
        out.append(
            {
                "set": name,
                "cell": cell,
                "n": int(group["n"].iloc[0]),
                "replicates": len(group),
                "standardized_bias": float(error.mean() / spread),
                "mcse": float(1.0 / np.sqrt(len(group))),
                "se_ratio": float(group["std_error"].mean() / spread),
                "coverage": float(np.mean(np.abs(error) <= half)),
                "covered": float(group["covered"].mean()) if "covered" in group else np.nan,
                "rejected": float(group["rejected"].mean()) if "rejected" in group else np.nan,
                "bias": float(error.mean()),
                "bias_se": float(error.std(ddof=1) / np.sqrt(len(group))),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--set", action="append", choices=sorted(SETS), help="rerun one set")
    args = parser.parse_args()
    names = args.set or list(SETS)
    payloads = [(name, replicate) for name in names for replicate in range(SETS[name][5])]
    outcomes = map_parallel(_fit, payloads, n_jobs=args.jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    if args.set:
        kept = pd.read_csv(HERE / "rows.csv.gz")
        kept["n"] = kept["n"].fillna(N).astype(int) if "n" in kept else N
        rows = pd.concat([kept[~kept["set"].isin(names)], rows], ignore_index=True)
    rows.to_csv(HERE / "rows.csv.gz", index=False, lineterminator="\n", float_format="%.17g")
    reading(rows).to_csv(
        HERE / "reading.csv", index=False, lineterminator="\n", float_format="%.6g"
    )


if __name__ == "__main__":
    main()
