"""The SE shortfall of the logit-MSM slope in the smallest stratum of the stratified study.

``canonical-stratified-incremental-msm`` published two red calibration cells:
``logit_v1_a__correctly_specified`` and ``logit_v2_a__correctly_specified``.  This diagnostic
asks whether the shortfall comes from the stratified construction.  It fits two arms on fresh
draws, seeded apart from every declared stream:

=========  =====================================================================================
arm        fit
=========  =====================================================================================
pooled     the declared stratified logit-MSM fit of L1 at total size ``n``
subset     the shipped unstratified logit-MSM fit on a sample of the law given ``V = 2``, of the
           size stratum 2 has inside the pooled fit (``0.2 n``)
=========  =====================================================================================

Each arm runs at the declared size and at four times it.  The rows go to
``rows.csv.gz``.  ``tests/unit/test_band_shortfall_reading.py`` rebuilds the reading from them.

    python -m tests.diagnostics.x8_logit_small_stratum.run --jobs 16
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.studies import stratified_alternating_law as law
from tests.studies import stratified_incremental_msm_properties as properties
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent
STRATUM = 2
#: Each run: the arm, the total size ``n`` and the number of replications.
DESIGN: tuple[tuple[str, int, int], ...] = (
    ("pooled", 2_000, 2_000),
    ("subset", 2_000, 2_000),
    ("pooled", 8_000, 1_000),
    ("subset", 8_000, 1_000),
)


def _seed(arm: str, n: int, replicate: int) -> int:
    return stream_seed(properties.STUDY, "diagnostic", "x8_logit_small_stratum", arm, n, replicate)


def _fit(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    arm, n, replicate = payload
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if arm == "pooled":
            result = properties.fit_msm(law.sample(n, _seed(arm, n, replicate)))
            name = f"msm[a][V={STRATUM}]"
        else:
            size = round(n * float(law.base.P_V[STRATUM]))
            frame = law.sample(8 * n, _seed(arm, n, replicate))
            frame = frame[frame["V"] == STRATUM].iloc[:size].reset_index(drop=True)
            if len(frame) < size:  # pragma: no cover - a design guard
                raise RuntimeError("the draw held too few stratum rows")
            result = (
                TMLE(
                    msm=properties.L1_MSM,
                    outcome_learner=properties.logistic(),
                    treatment_learner=properties.logistic(),
                    cross_fit=False,
                    simultaneous=False,
                    max_iter=100,
                    tol=1e-10,
                    random_state=0,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W"])
                .single()
            )
            name = "msm[a]"
    estimate = result[name]
    return [
        {
            "arm": arm,
            "n": n,
            "replicate": replicate,
            "truth": law.TRUTH_MSM["logit"][f"msm[a][V={STRATUM}]"],
            "estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()
    payloads = [((arm, n, r),) for arm, n, reps in DESIGN for r in range(reps)]
    outcomes = map_parallel(_fit, payloads, n_jobs=arguments.jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    with open(HERE / "rows.csv.gz", "wb") as handle:
        rows.to_csv(
            handle, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n"
        )
    print(rows.groupby(["arm", "n"]).apply(_summary).to_string())


def _summary(group: pd.DataFrame) -> pd.Series:
    spread = float(group["estimate"].std(ddof=1))
    return pd.Series(
        {
            "replicates": len(group),
            "se_ratio": float(group["std_error"].mean()) / spread,
            "standardized_bias": float((group["estimate"] - group["truth"]).mean()) / spread,
        }
    )


if __name__ == "__main__":
    main()
