"""The red ``binary_lasso`` coverage row of ``canonical-ctmle-sequences``, refitted beside TMLE.

The registered row reads a coverage of 0.92375 with a 99% interval of 0.896 to 0.946 against a
floor of 0.90, and an SE ratio of 0.905.  The plan's probe expected the red (coverage 0.899,
SE ratio 0.881).  This diagnostic asks whether the shortfall belongs to the lasso selection or
to the plug-in variance on the law.  On each of the 800 registered samples it refits the
study's lasso C-TMLE and plain TMLE with the same main-terms logistic learners and the same
propensity bounds, ``study.G_BOUNDS``.

Each row records the lasso estimate, its plug-in standard error, the reported candidate and
whether it is the penalty 0, the TMLE estimate and standard error, and the share of the true
propensity outside the bounds on that sample.  The diagnostic cannot change the row's verdict.

The rows go to ``rows.csv.gz`` and the reading to ``reading.csv``.
``tests/unit/test_x16_lasso_coverage_reading.py`` rebuilds the reading from the rows and checks
the lasso rows against the committed replicates.

    python -m tests.diagnostics.x16_lasso_coverage.run --jobs 4
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.studies import canonical_ctmle_sequences as study

HERE = Path(__file__).resolve().parent
SCENARIO = "binary_lasso"
CRITICAL = 1.959964
#: The 0-based index of the penalty 0, the last candidate of the lasso path.
UNPENALIZED = study.PATH_LENGTHS[SCENARIO] - 1


def _fit(replicate: int) -> dict[str, float | int]:
    frame, truth = study.draw_scenario(SCENARIO, study.PRIMARY_N, replicate)
    covariates = list(study.COVARIATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = study.fit_cleverly(frame, SCENARIO, replicate)
        plain = (
            TMLE(
                outcome_learner=study._logistic(),
                treatment_learner=study._logistic(),
                cross_fit=False,
                estimands=("ate",),
                g_bounds=study.G_BOUNDS,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=covariates)
            .single()
        )
    selection = lasso.extra["ctmle"]
    g0 = study.scenario_law(SCENARIO).propensity(frame[covariates].to_numpy())
    low, high = study.G_BOUNDS
    return {
        "replicate": replicate,
        "truth": float(truth["ate"]),
        "lasso_estimate": float(lasso["ate"].psi),
        "lasso_std_error": float(lasso["ate"].plugin_std_error),
        "lasso_reported": int(selection.reported),
        "lasso_unpenalized": int(selection.reported == UNPENALIZED),
        "tmle_estimate": float(plain["ate"].psi),
        "tmle_std_error": float(plain["ate"].std_error),
        "truncated_share": float(np.mean((g0 < low) | (g0 > high))),
    }


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """Coverage and SE ratio of each estimator, and both in the lowest standard-error quartile."""
    out = []
    for name in ("lasso", "tmle"):
        estimate = rows[f"{name}_estimate"].to_numpy()
        error = rows[f"{name}_std_error"].to_numpy()
        covered = np.abs(estimate - rows["truth"].to_numpy()) <= CRITICAL * error
        lowest = error <= np.quantile(error, 0.25)
        out.append(
            {
                "estimator": name,
                "replicates": len(rows),
                "coverage": float(covered.mean()),
                "se_ratio": float(error.mean()) / float(estimate.std(ddof=1)),
                "lowest_quartile_coverage": float(covered[lowest].mean()),
                "unpenalized": int(rows["lasso_unpenalized"].sum()) if name == "lasso" else -1,
                "truncated_share": float(rows["truncated_share"].mean()),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    replicates = list(range(study.PRIMARY_REPLICATES))
    rows = pd.DataFrame(map_parallel(_fit, replicates, n_jobs=arguments.jobs))
    rows = rows.sort_values("replicate", ignore_index=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        arguments.output / "rows.csv.gz", index=False, float_format="%.17g", lineterminator="\n"
    )
    reading(rows).to_csv(
        arguments.output / "reading.csv", index=False, float_format="%.6g", lineterminator="\n"
    )


if __name__ == "__main__":
    main()
