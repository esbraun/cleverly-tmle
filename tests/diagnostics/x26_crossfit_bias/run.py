r"""The second-order corrected plug-in omitted-variable bound, on the binary law of X26.

A design-review pilot found the uncorrected cross-fitted ATE :math:`\nu^2` limit covering 0.887
against a band of 0.92 to 0.98: the plug-in :math:`E_n[\hat\alpha^2]` carries a bias of order
:math:`1/n`, larger out of fold.  The package now subtracts the second-order bias
:math:`\hat B` of the stacked estimating equation (``cleverly.sensitivity._second_order``) and
corrects the variance estimate of :math:`\hat\nu^2` in the same way.  This diagnostic reads both
estimates, in sample and cross-fitted, on the same draws.

On the binary law of ``tests.studies.omitted_variable_bound_plugin`` (``make_linear_ate``) at
strength ``cf_y = cf_d = 0.5`` and ``rho = 1``, each draw is fitted twice.

``crossfit``
    ``TMLE(cross_fit=True, n_folds=5, q_bounds=(-40.0, 45.0))`` with a ``LinearRegression``
    outcome and ``LogisticRegression(C=numpy.inf, max_iter=2000)``, estimands ATE, ATT and ATC,
    ``random_state=0``.
``insample``
    The registered ``binary_logistic`` fit, ``cross_fit=False``, on the same draw.

Each row records four quantities, each with its reported standard error.

``nu2_raw``
    :math:`E_n[\hat\alpha^2]`, with ``sqrt(influence_variance(psi_nu2))``.
``nu2``
    The corrected :math:`E_n[\hat\alpha^2] - \hat B`, with the root of the corrected variance.
``lower``, ``upper``
    The bound ends, with the standard error read back off the reported limit.

A replicate the package refuses keeps ``nu2_raw`` and records the refusal key.  Its corrected
quantities are missing, and the reading counts the refusals.

**The reading.**  Per n, fitting, parameter and row, ``reading.csv`` gives the coverage of the
95% interval with its 99% Wilson interval, the bias in standard deviations of the estimate, the
ratio of the mean reported standard error to that deviation, and ``refused``.  The corrected rows
read the replicates the package did not refuse.  The raw row reads all of them.  The
measurement: the raw cross-fitted ATE ``nu2`` covers 0.8845 with bias 1.44 SD at n = 1000.  The
corrected ``nu2`` has bias within 0.1 SD in every cell, and its SE ratio is 0.73 to 0.75
cross-fitted and 0.90 to 0.92 in sample at n = 1000.  The corrected bound rows cover 0.947 to
0.959 with SE ratio 0.986 to 1.019 in every cell.

**Seeds.**  The draw of replicate ``k`` at sample size ``n`` uses
``stream_seed(STUDY, "x26-crossfit-bias", n, k)``.  That stream hashes a label into the
spawn key of the study's resampling seed.  The registered study draws with
``replicate_seed``, which spawns on the study seed and an integer scenario index, so the two
streams are disjoint.  Each ``n`` has its own stream.

**The n ladder.**  2,000 replicates at ``n = 1000`` and 1,000 at ``n = 4000``.

The rows go to ``rows.csv.gz`` and the reading to ``reading.csv``.
``tests/unit/test_x26_crossfit_bias_reading.py`` rebuilds the reading from the rows.

    python -m tests.diagnostics.x26_crossfit_bias.run --jobs 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import omitted_variable as ov
from cleverly.utils.parallel import map_parallel
from tests.studies import omitted_variable_bound_plugin as study
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).resolve().parent

#: The scenario whose law, truths and in-sample fit the diagnostic reuses.
SCENARIO = "binary_logistic"
#: The replicates at each sample size: the n ladder.
ARMS: dict[int, int] = {1_000: 2_000, 4_000: 1_000}
#: The declared outcome support of the cross-fitted fit, as the registered cross-fit scenarios declare it.
Q_BOUNDS = (-40.0, 45.0)
FITTINGS = ("crossfit", "insample")
PARAMETERS = tuple(study.BINARY_PARAMETERS)
ROWS = ("nu2_raw", "nu2", "lower", "upper")
#: The truth each row estimates.
TRUTH_OF = {"nu2_raw": "nu2", "nu2": "nu2", "lower": "lower", "upper": "upper"}
CRITICAL = 1.959964
LABEL = "x26-crossfit-bias"


def seed_of(n: int, replicate: int) -> int:
    """The seed of replicate ``replicate`` at sample size ``n``, distinct from the study's."""
    return stream_seed(study.STUDY, LABEL, n, replicate)


def fit_crossfit(frame: pd.DataFrame) -> object:
    """The cross-fitted fit that the removed ``binary_logistic_crossfit`` scenario declared."""
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(C=np.inf, max_iter=2000),
            cross_fit=True,
            n_folds=5,
            q_bounds=Q_BOUNDS,
            simultaneous=False,
            estimands=("ate", "att", "atc"),
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def parameter_values(result: object, token: str) -> dict[str, float | str]:
    """Each row's estimate and reported standard error, and the refusal key or ``""``."""
    name = study.PARAMETERS[SCENARIO][token][0]
    elements = ov.sensitivity_elements(result, name, nu2_estimator="plugin")  # type: ignore[arg-type]
    if elements.psi_nu2 is None:
        raise RuntimeError(f"the plug-in bound of {name!r} has no curve")
    cluster = result.data.cluster  # type: ignore[attr-defined]
    correction = 0.0 if elements.nu2_correction is None else elements.nu2_correction
    out: dict[str, float | str] = {
        "nu2_raw": float(elements.nu2) + correction,
        "nu2_raw_se": float(np.sqrt(influence_variance(elements.psi_nu2, cluster))),
        "refusal": "" if elements.limit_refusal is None else elements.limit_refusal.key,
    }
    for row in ("nu2", "lower", "upper"):
        out[row] = out[f"{row}_se"] = math.nan
    if elements.limit_refusal is not None:
        return out
    assert elements.nu2_variance_correction is not None
    bounds = ov.omitted_variable_bounds(
        result,  # type: ignore[arg-type]
        name,
        cf_y=study.CF_Y,
        cf_d=study.CF_D,
        rho=study.RHO,
        nu2_estimator="plugin",
    )
    variance = influence_variance(elements.psi_nu2, cluster) - elements.nu2_variance_correction
    out["nu2"] = float(elements.nu2)
    out["nu2_se"] = float(np.sqrt(variance))
    out["lower"] = float(bounds.lower)
    out["lower_se"] = float((bounds.lower - bounds.ci_lower) / study.ONE_SIDED)
    out["upper"] = float(bounds.upper)
    out["upper_se"] = float((bounds.ci_upper - bounds.upper) / study.ONE_SIDED)
    return out


def replicate_rows(n: int, replicate: int) -> list[dict[str, float | int | str]]:
    """The rows of one draw: both fittings, each parameter."""
    frame, truth = study.draw_from_seed(SCENARIO, n, seed_of(n, replicate))
    out: list[dict[str, float | int | str]] = []
    results = {"crossfit": fit_crossfit(frame), "insample": study.fit(frame, SCENARIO)}
    for fitting, result in results.items():
        for token in PARAMETERS:
            row: dict[str, float | int | str] = {
                "n": n,
                "replicate": replicate,
                "fitting": fitting,
                "parameter": token,
                **parameter_values(result, token),
            }
            for name in ROWS:
                row[f"{name}_truth"] = float(truth[study.estimand_key(token, TRUTH_OF[name])])
            out.append(row)
    return out


def _replicate(n: int, replicate: int) -> list[dict[str, float | int | str]]:
    return replicate_rows(n, replicate)


def wilson(covered: int, total: int, level: float = 0.99) -> tuple[float, float]:
    """The Wilson score interval of a proportion."""
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    p = covered / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """Per n, fitting, parameter and row: coverage, its 99% Wilson interval, bias and SE ratio.

    ``bias_sd`` is the mean estimate minus the truth over the empirical standard deviation of
    the estimate.  ``se_ratio`` is the mean reported standard error over that deviation.  The
    corrected rows read the replicates the package did not refuse; ``refused`` counts the rest.
    """
    out = []
    for n in ARMS:
        for fitting in FITTINGS:
            for token in PARAMETERS:
                cell = rows.loc[
                    (rows["n"] == n) & (rows["fitting"] == fitting) & (rows["parameter"] == token)
                ]
                refused = int(cell["refusal"].fillna("").astype(str).ne("").sum())
                for name in ROWS:
                    kept = cell.loc[cell[name].notna()]
                    estimate = kept[name].to_numpy()
                    error = kept[f"{name}_se"].to_numpy()
                    truth = float(cell[f"{name}_truth"].iloc[0])
                    covered = int(np.sum(np.abs(estimate - truth) <= CRITICAL * error))
                    low, high = wilson(covered, len(kept))
                    spread = float(estimate.std(ddof=1))
                    out.append(
                        {
                            "n": n,
                            "fitting": fitting,
                            "parameter": token,
                            "row": name,
                            "replicates": len(kept),
                            "refused": 0 if name == "nu2_raw" else refused,
                            "coverage": covered / len(kept),
                            "coverage_lower99": low,
                            "coverage_upper99": high,
                            "bias_sd": (float(estimate.mean()) - truth) / spread,
                            "se_ratio": float(error.mean()) / spread,
                        }
                    )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    payloads = [(n, k) for n, count in ARMS.items() for k in range(count)]
    records = map_parallel(_replicate, payloads, n_jobs=arguments.jobs)
    rows = pd.DataFrame([row for batch in records for row in batch])
    rows["fitting"] = pd.Categorical(rows["fitting"], FITTINGS, ordered=True)
    rows = rows.sort_values(["n", "replicate", "fitting", "parameter"], ignore_index=True)
    rows["fitting"] = rows["fitting"].astype(str)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        arguments.output / "rows.csv.gz", index=False, float_format="%.17g", lineterminator="\n"
    )
    reading(rows).to_csv(
        arguments.output / "reading.csv", index=False, float_format="%.6g", lineterminator="\n"
    )


if __name__ == "__main__":
    main()
